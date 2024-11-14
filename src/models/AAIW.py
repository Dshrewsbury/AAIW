import math
from typing import Any

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import MaxMetric, MeanMetric

from src.models.components import clip
from src.models.components.SemanticDecoupling import TransformerSemanticDecoupling
from src.models.components.clip_vit import CLIPVIT
from src.models.components.losses import consistency_loss, SPLC, AsymmetricLoss, Hill
from src.models.components.utils import convert_models_to_fp32, compute_metrics, calculate_instance_entropy, \
    param_groups_lrd


class AAIWModule(LightningModule):

    def __init__(
            self,
            loss_function: 'bce-l',
            fix_layer: 10,
            weight_decay: 0.05,
            layer_decay: 0.65,
            lr: 2e-3,
            min_lr: 1e-6,
            localpatch_topk: 16,
            warmup_epochs: 2,
            epochs: 20,
            cos_topk: 15,
            creg_topk: 15,
            creg_delta: 0.9,
            example_num: 100,
            num_classes: 20,
            dataset: 'pascal',
            c_scale=1.5
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Settings
        self.loss_function = loss_function
        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.c_scale = c_scale

        # Dataset-specific label embeddings
        self.label_embed = self._load_label_embeddings(dataset)

        # Model components
        clip_model, _ = clip.load('ViT-B/16', jit=False)
        convert_models_to_fp32(clip_model)
        self.model = CLIPVIT(localpatch_topk, clip_model, self.device)
        convert_models_to_fp32(self.model)

        # Loss function setup
        self.criterion = self._initialize_loss_function(loss_function)

        # Metrics
        self.train_loss = MeanMetric()
        self.reg_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.map_val = compute_metrics
        self.map_test = compute_metrics
        self.map_val_best = MaxMetric()

        # Logging
        self.iter = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Feature management
        self.pos_feature = torch.zeros((num_classes, example_num, 512))
        self.pos_prediction = torch.zeros((num_classes, example_num, 1))
        self.bank_ptrs = torch.zeros(num_classes, dtype=torch.long)
        self.example_num = example_num
        self.cos_topk = cos_topk
        self.creg_delta = creg_delta
        self.creg_topk = creg_topk
        self.max_entropy = 0
        self.min_entropy = 0

        # Ambiguity and semantic decoupling
        self.semantic_decoupling = TransformerSemanticDecoupling(self.label_embed)

        # Freeze layers in mode
        self._freeze_model_layers(fix_layer)


    def _load_label_embeddings(self, dataset):
        """Load label embeddings based on the dataset."""
        if dataset == 'pascal':
            return torch.load("./src/models/components/pascal_label_embeddings.pt", map_location=self.device).to(
                torch.float32)
        elif dataset == 'coco':
            return torch.load("./src/models/components/coco_label_embeddings.pt", map_location=self.device).to(
                torch.float32)
        elif dataset == 'vg':
            return './src/models/components/vg_label_embeddings.pt'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")


    def _initialize_loss_function(self, loss_function):
        """Initialize the appropriate loss function."""
        if loss_function == 'bce-l':
            return torch.nn.BCEWithLogitsLoss(reduction='none')
        elif loss_function == 'asymm':
            return AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.2, disable_torch_grad_focal_loss=True)
        elif loss_function == 'bce':
            return torch.nn.BCEWithLogitsLoss()
        elif loss_function == 'splc':
            return SPLC(tau=0.6, change_epoch=1, margin=1, gamma=2, reduction='sum')
        elif loss_function == 'hill':
            return Hill(lamb=1.5, margin=1.0, gamma=2.0, reduction='sum')
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")


    def _freeze_model_layers(self, fix_layer):
        """Freeze specific layers in the model."""
        for name, param in self.model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad_(False)


    def forward(self, image):
        label_embed = self.label_embed.to(self.device)
        return self.model(image, label_embed, self.device)


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.map_val_best.reset()
        self.dataset_length = (len(self.trainer.train_dataloader.dataset))

        self.confident_pseudolabels = torch.zeros((self.dataset_length, self.num_classes),
                                                  dtype=torch.float32)

        self.selected_label = torch.zeros(
            (self.dataset_length, self.num_classes),
            dtype=torch.long,
        )
        self.selected_label = self.selected_label.to(self.device)


    def training_step(self, batch: Any, batch_idx: int):

        # get image and text features
        images, strong_images, partial_labels, targets, indices = batch

        # Step through model with weak image and get "predictions"
        logits, pred_feat, dist_feat = self.forward(images)
        preds = logits.clamp_min(0)
        semantic_features = self.semantic_decoupling(pred_feat, self.device)

        self.update_feature(semantic_features, partial_labels, preds)
        entropy_weight, instance_entropy = self.ambiguity_weighting(semantic_features, preds, partial_labels)
        entropy_weight = entropy_weight + (1 - entropy_weight) * (
                    1 - math.exp(-self.c_scale * (self.current_epoch / self.epochs)))

        if self.current_epoch == 0:
            entropy_weight = None

        loss = 0
        splc_loss = 0

        # Consistency Regularization
        s_logits, _, _ = self.forward(strong_images)
        reg_loss = self.consistency_regularization(s_logits, logits, indices, entropy_weight)

        # Loss Calculation
        if self.loss_function == 'bce-l':
            mask = partial_labels == 1  # filters out both unknown and negative labels
            masked_labels = partial_labels * mask
            label_level_loss = self.criterion(logits, masked_labels.float())
            masked_label_level_loss = label_level_loss * mask.float()
            loss += masked_label_level_loss.sum(dim=1).mean()
        elif self.loss_function == 'splc':
            partial_labels.clamp_min_(0)
            splc_loss, label_level_loss = self.criterion(logits, partial_labels.float(), self.current_epoch,
                                                         entropy_weight)
        else:
            loss += self.criterion(logits, partial_labels.float())

        loss = splc_loss + reg_loss

        # Logging
        self.train_loss(loss)
        self.reg_loss(reg_loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/reg_loss", self.reg_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/splc_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        output = {"all_targets": targets.cpu().detach(), "entropy_weight": instance_entropy.cpu().detach()}
        self.training_step_outputs.append(output)

        return loss


    def validation_step(self, batch: Any, batch_idx: int):

        images, _, _, targets, indices = batch
        logits, pred_feat, dist_feat = self.forward(images)
        preds = torch.sigmoid(logits)

        if self.loss_function == 'splc':
            loss, label_level_loss = self.criterion(logits, targets, self.current_epoch, None)
        else:
            loss = self.criterion(logits, targets)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        output = {"preds": preds.cpu().detach(), "labels": targets.cpu().detach(), "logits": logits.cpu().detach()}
        self.validation_step_outputs.append(output)


    def on_validation_epoch_end(self):

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)

        map_val = self.map_val(all_preds.detach(), all_labels.numpy())
        map_val = map_val['map']

        self.log("val/map", map_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.map_val_best(map_val)
        self.log("val/map_best", self.map_val_best.compute(), prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()


    def test_step(self, batch: Any, batch_idx: int):

        images, _, _, targets, indices = batch
        logits, pred_feat, dist_feat = self.forward(images)

        preds = torch.sigmoid(logits)

        # update and log metrics
        output = {"preds": preds.cpu().detach(), "labels": targets.cpu().detach(), "logits": logits.cpu().detach()}
        self.test_step_outputs.append(output)


    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0)
        map_test = self.map_test(all_preds.detach(), all_labels.numpy())
        map_test = map_test['map']
        self.log("test/map", map_test, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def configure_optimizers(self):

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader
        params, param_group_names = param_groups_lrd(self.model, self.hparams.fix_layer, self.hparams.weight_decay,
                                                     layer_decay=self.hparams.layer_decay
                                                     )

        params_name = []
        for k, v in param_group_names.items():
            params_name += v["params"]

        for name, param in self.model.named_parameters():
            if name not in params_name:
                param.requires_grad = False

        optimizer = optim.Adam(params=params, lr=self.hparams.lr, weight_decay=0)
        scheduler = OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=int(len(dataloader)),
                               epochs=self.trainer.max_epochs,
                               pct_start=0.2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    def consistency_regularization(self, s_logits, logits, indices, entropy_weight):

        pseudo_counts = self.confident_pseudolabels.sum(dim=0)  # Summing the presence of each label.

        # Count instances with no predicted pseudo-labels
        no_label_count = (self.confident_pseudolabels.sum(dim=1) == 0).float().sum()

        if no_label_count != self.dataset_length:
            if self.current_epoch < self.warmup_epochs:
                norm_class_freq = pseudo_counts / no_label_count
            else:
                # Normalize by the most frequently predicted label
                norm_class_freq = pseudo_counts / (pseudo_counts.max())  # + 1e-5)
        else:
            # In case no instances have labels yet, set classwise_acc to zeros
            norm_class_freq = torch.zeros_like(pseudo_counts)

        creg_loss, confident_labels, confident_label_indices = consistency_loss(logits, s_logits, self.device,
                                                                                self.creg_topk, norm_class_freq)
        lambda_cst_tensor = torch.full_like(creg_loss, self.creg_delta)

        if entropy_weight is not None:
            creg_loss = (entropy_weight * creg_loss).mean()
        else:
            creg_loss = (lambda_cst_tensor * creg_loss).mean()

        for i, idx in enumerate(indices):
            # Find which labels are confident for the current instance
            confident_indices_for_instance = confident_label_indices[i][confident_labels[i] == 1]

            # Update selected_label for those confident labels
            self.confident_pseudolabels[idx, confident_indices_for_instance] = 1

        return creg_loss


    def ambiguity_weighting(self, feature, preds, partial_labels):

        with torch.no_grad():
            missing_indices = torch.nonzero(partial_labels == -1, as_tuple=True)
            batch_size, class_num, feature_dim = feature.size()
            example_num = self.pos_feature.size(1)
            cos = torch.nn.CosineSimilarity(dim=2, eps=1e-9)

            self.pos_feature = self.pos_feature.to(self.device)
            self.pos_prediction = self.pos_prediction.to(self.device)

            # feature1: features for the missing labels in the batch
            # feature2: missing label feature's corresponding positively labeled features for same class
            feature1 = feature[partial_labels == -1].view(-1, 1, feature_dim).repeat(1, example_num,
                                                                                    1)  # missing labels, exampleNum, feature size]
            missing_label_indices = missing_indices[1].to(self.device)
            feature2 = self.pos_feature[missing_label_indices]  # [missing labels, exampleNum, feature size]
            neighbor_pred = self.pos_prediction[missing_label_indices]

            cosine_similarities = cos(feature1, feature2)  # [number of missing labels, exampleNum]
            topk_values, topk_indices = torch.topk(cosine_similarities, k=self.cos_topk, dim=1)

            # Get the corresponding top neighbor predictions
            rows = torch.arange(topk_indices.size(0)).view(-1, 1).expand_as(topk_indices).reshape(-1)
            cols = topk_indices.reshape(-1)
            topk_preds = neighbor_pred[rows, cols].view(*topk_indices.size())
            mean_topk_predictions = torch.mean(topk_preds.float(), dim=1)

            # Replace missing label locations in the copied predictions tensor with the computed mean_topk_predictions
            uncertain_preds = preds.clone()
            uncertain_preds[missing_indices] = mean_topk_predictions
            instance_entropy = calculate_instance_entropy(uncertain_preds)

            normalized_entropy = (instance_entropy - self.min_entropy) / (self.max_entropy - self.min_entropy)
            w = torch.exp(-torch.tensor(normalized_entropy, device=self.device))

            return w, instance_entropy


    def update_feature(self, feature, target, predictions):
        feature = feature.detach().clone()

        for c in range(self.num_classes):
            pos_feature = feature[:, c][target[:, c] == 1]
            pos_pred = predictions[:, c][target[:, c] == 1].view(-1, 1)
            num_new_samples = pos_feature.shape[0]

            # Loop over new samples
            for i in range(num_new_samples):
                # Use the pointer to decide where to insert the new data
                insert_idx = self.bank_ptrs[c] % self.example_num

                self.pos_feature[c, insert_idx] = pos_feature[i]
                self.pos_prediction[c, insert_idx] = pos_pred[i]

                # Increment the pointer
                self.bank_ptrs[c] += 1


    @staticmethod
    def curriculum_scaling(epoch, total_epochs, base=0.1, scale=3):
        progression = epoch / total_epochs  # Linear progression between 0 and 1
        scaling_factor = base ** (-scale * progression)  # Exponential scaling
        return scaling_factor


if __name__ == "__main__":
    _ = AAIWModule(None, None, None)
