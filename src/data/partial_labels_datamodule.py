import os
import random
from typing import Optional

import numpy as np
from PIL import Image
from PIL import ImageDraw
from pytorch_lightning import LightningDataModule
from randaugment import RandAugment
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.data.components.noise_generation_ml import *


def get_base_path():
    # This gives you the directory containing the script you're currently running
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # This gives you the directory two levels up from the current script's directory
    parent_dir = os.path.dirname(os.path.dirname(current_script_dir))

    return parent_dir


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor


    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def generate_split(num_ex, frac, rng):
    # Compute size of each split
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2

    # Assign indices to splits
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[n_1:])

    return idx_1, idx_2


class PartialMultiLabel(Dataset):

    def __init__(self,
                 image_names,
                 partial_labels,
                 true_labels,
                 data_dir="",
                 transforms=None,
                 strong_transforms=None):

        self.image_names = image_names
        self.ground_truth_labels = true_labels
        self.data_dir = data_dir
        self.transforms = transforms
        self.strong_transforms = strong_transforms
        if partial_labels is None:
            self.labels_pt = true_labels
        else:
            self.labels_pt = partial_labels


    def get_all_images(self):
        images = []
        for i in range(len(self.image_names)):
            image_path = os.path.join(self.data_dir, self.image_names[i])
            image = Image.open(image_path).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
            images.append(image)
        return torch.stack(images)


    def __len__(self):
        return len(self.image_names)


    # returns the now noisy target
    def __getitem__(self, index):

        image_path = os.path.join(self.data_dir, self.image_names[index])
        with Image.open(image_path) as I_raw:
            image = I_raw.convert('RGB')

        if self.transforms is not None:
            weak_image = self.transforms(image)

        if self.strong_transforms is not None:
            strong_image = self.strong_transforms(image)
        else:
            strong_image = None

        return weak_image, strong_image, self.labels_pt[index, :], self.ground_truth_labels[index, :], index


class PartialMLDataModule(LightningDataModule):

    def __init__(
            self,
            dataset_name: str = 'pascal',
            partial_mode: str = 'SP',
            num_classes: int = 80,
            batch_size: int = 32,
            num_workers: int = 8,
            label_proportion: float = 0.6,
            val_frac: float = 0.2,
            split_seed: int = 42,
            ss_frac_val: float = 1.0,
            ss_frac_train: float = 1.0,
            ss_seed: int = 1200,
            pin_memory: bool = False,
            train_transformations=None,
            val_transformations=None,
            test_transformations=None,
            strong_transformations=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.partial_mode = partial_mode
        if isinstance(self.partial_mode, list):
            self.partial_mode = self.partial_mode[0]
        self.dataset_name = dataset_name
        self.val_frac = val_frac
        self.split_seed = split_seed
        self.ss_frac_val = ss_frac_val
        self.ss_frac_train = ss_frac_train
        self.ss_seed = ss_seed
        self.label_proportion = label_proportion

        base_path = get_base_path()
        if dataset_name == 'pascal':
            self.image_dir = os.path.join(base_path, 'data', 'voc', 'VOCdevkit', 'VOC2012', 'JPEGImages')
            self.anno_dir = os.path.join(base_path, 'data', 'voc')
            self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                               'train', 'tvmonitor']
        elif dataset_name == 'coco':
            self.image_dir = os.path.join(base_path, 'data/coco/')
            self.anno_dir = os.path.join(base_path, 'data/coco/annotations/2014')
            self.classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                               "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                               "kite",
                               "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                               "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                               "orange",
                               "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                               "potted plant",
                               "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                               "cell phone",
                               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                               "scissors",
                               "teddy bear", "hair drier", "toothbrush"]
        else:
            raise NotImplementedError('Dataset not supported.')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        img_size = 224
        if val_transformations is None:
            self.transform_train = transforms.Compose([
                # transforms.RandomResizedCrop(img_size)
                transforms.Resize((img_size, img_size)),
                CutoutPIL(cutout_factor=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.transform_val = transforms.Compose([
                # transforms.CenterCrop(img_size),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            self.transform_test = self.transform_val

            self.transform_strong = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])


        else:
            self.transform_train = train_transformations
            self.transform_val = val_transformations
            self.transform_test = test_transformations

            if strong_transformations is None:
                self.transform_strong = train_transformations
            else:
                self.transform_strong = strong_transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        data = {}
        for phase in ['train', 'val']:
            data[phase] = {}
            data[phase]['labels'] = np.load(os.path.join(self.anno_dir, 'processed_{}_labels.npy'.format(phase)))

            # probably need to refactor this as having a separate partial_labels variable doesnt work
            if self.partial_mode == "normal":
                data[phase]['partial_labels'] = None
            else:
                if self.partial_mode in ['random', 'idn']:
                    data[phase]['partial_labels'] = np.load(os.path.join(self.anno_dir,
                                                                         f'partial_{self.partial_mode}_{int(self.label_proportion * 100)}_{phase}_labels.npy'))
                else:
                    data[phase]['partial_labels'] = np.load(
                        os.path.join(self.anno_dir, f'partial_{self.partial_mode}_{phase}_labels.npy'))

            data[phase]['images'] = np.load(os.path.join(self.anno_dir, 'processed_{}_images.npy'.format(phase)))

        # generate indices to split official train set into train and val:
        split_idx = {'train': (generate_split(
            len(data['train']['images']),
            self.val_frac,
            np.random.RandomState(self.split_seed)
        ))[0], 'val': (generate_split(
            len(data['train']['images']),
            self.val_frac,
            np.random.RandomState(self.split_seed)
        ))[1]}

        # subsample split indices:
        ss_rng = np.random.RandomState(self.ss_seed)
        for phase in ['train', 'val']:
            num_initial = len(split_idx[phase])

            # selects a fraction of the validation set/train set, maybe to set fraction of dataset to be sampled
            if phase == 'train':
                num_final = int(np.round(self.ss_frac_train * num_initial))
            else:
                num_final = int(np.round(self.ss_frac_val * num_initial))
            split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]

        # Load the metadata from npy/csv/w.e files, based on a provided parameter
        partial_labels = {}
        if self.partial_mode == 'normal':
            partial_labels['train'] = None
            partial_labels['val'] = None
        else:
            partial_labels['train'] = data['train']['partial_labels'][split_idx['train'], :]
            partial_labels['val'] = data['train']['partial_labels'][split_idx['val'], :]

        # Just import PartialMultilabel, read the files in and voila create dataloader
        self.data_train = PartialMultiLabel(image_names=data['train']['images'][split_idx['train']],
                                            partial_labels=partial_labels['train'],
                                            true_labels=data['train']['labels'][split_idx['train'], :],
                                            data_dir=self.image_dir,
                                            transforms=self.transform_train,
                                            strong_transforms=self.transform_strong)

        self.data_val = PartialMultiLabel(image_names=data['train']['images'][split_idx['val']],
                                          partial_labels=partial_labels['val'],
                                          true_labels=data['train']['labels'][split_idx['val'], :],
                                          data_dir=self.image_dir,
                                          transforms=self.transform_val,
                                          strong_transforms=self.transform_strong)

        self.data_test = PartialMultiLabel(image_names=data['val']['images'],
                                           partial_labels=data['val']['partial_labels'],
                                           true_labels=data['val']['labels'],
                                           data_dir=self.image_dir,
                                           transforms=self.transform_test,
                                           strong_transforms=self.transform_strong)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "pascal.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
