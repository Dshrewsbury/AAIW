from pathlib import Path
import pytest
import torch

from src.data import COCODataModule
from src.data import VOC2007DataModule
from src.data import VOC2012DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "/home/dan/LabelNoise/data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_COCO_datamodule(batch_size):
    data_dir = "/home/dan/LabelNoise/data/MultiLabel/coco/"

    dm = COCODataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val
    assert Path(data_dir, "annotations").exists()
    assert Path(data_dir, "train2014").exists()
    assert Path(data_dir, "val2014").exists()
    dm.setup()

    assert dm.data_train and dm.data_val
    assert dm.train_dataloader() and dm.val_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val)
    assert num_datapoints == 123_287

@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_voc2007_datamodule(batch_size):
    data_dir = "/home/dan/LabelNoise/data/MultiLabel/voc/"

    dm = VOC2007DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_train
    assert Path(data_dir, "VOCdevkit").exists()
    assert Path(data_dir, "files").exists()
    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_train
    assert dm.train_dataloader() and dm.val_dataloader()

    train_datapoints = len(dm.data_train)
    val_datapoints = len(dm.data_val)
    test_datapoints = len(dm.data_test)
    assert train_datapoints == 2_501
    assert val_datapoints == 2_510
    assert test_datapoints == 4_952

#
@pytest.mark.parametrize("batch_size", [32, 64, 128])
def test_voc2012_datamodule(batch_size):
    data_dir = "/home/dan/LabelNoise/data/MultiLabel/voc/"

    dm = VOC2012DataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_train
    assert Path(data_dir, "VOCdevkit").exists()
    assert Path(data_dir, "files").exists()
    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_train
    assert dm.train_dataloader() and dm.val_dataloader()

    train_datapoints = len(dm.data_train)
    val_datapoints = len(dm.data_val)
    test_datapoints = len(dm.data_test)
    assert train_datapoints == 5_717
    assert val_datapoints == 5_823
    assert test_datapoints == 4_952
