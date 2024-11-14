import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ranking_lossT(logitsT, labelsT):
    # Refer: https://github.com/akshitac8/BiAM

    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT), dim=0)
    subset_idxT = torch.nonzero(subset_idxT > 0).view(-1).long().cuda()
    sub_labelsT = labelsT[:, subset_idxT]
    sub_logitsT = logitsT[:, subset_idxT]
    positive_tagsT = torch.clamp(sub_labelsT, 0., 1.)
    negative_tagsT = torch.clamp(-sub_labelsT, 0., 1.)
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1)
    pos_score_matT = sub_logitsT * positive_tagsT
    neg_score_matT = sub_logitsT * negative_tagsT
    IW_pos3T = pos_score_matT.unsqueeze(1)
    IW_neg3T = neg_score_matT.unsqueeze(-1)
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0)
    violationT = torch.sign(diffT).sum(1).sum(1)
    diffT = diffT.sum(1).sum(1)
    lossT = torch.mean(diffT / (violationT + eps))

    return lossT


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def find_confident_labels(p_weak, T, device, K):
    # dim=1 = select the top confident labels for each instance?
    topk_values, topk_indices = torch.topk(p_weak, K, dim=1)
    topk_values = topk_values.to(device)
    # T = T.to(device)

    # Might want class-based thresholding
    # selected_T = T[topk_indices]
    selected_T = 0.5

    threshold_mask = topk_values > selected_T
    confident_labels = topk_indices[threshold_mask]
    return confident_labels, topk_indices, threshold_mask


# T is 0.5 for now
"""
 Promote consistency between predictions of weak/strongly augmented images
 We want to use the labels the model is most confident about per instance, as
 these should be used to guide the predictions of the strongly augmented version
 
 Though right now I dont think im doing what the paper did
"""

BCELoss = torch.nn.BCEWithLogitsLoss(reduction='none')


# not supposed to be logits, for the weak transformation
# but looks like its supposed to be logits for strong?
def consistency_loss(p_weak, p_strong, device, K, norm_class_freq):
    # Some logits are negative in CLIP
    p_weak = p_weak.clamp_min(0)
    #p_strong = p_strong.clamp_min(0)

    """
    For each instance they store the label that was predicted for it
    Then for each label they get the number of times it was predicted.

    Then their "classwise acc" they calculate depends on the warmup period, during warmup for each label they divide it by
    the maximum number of times a label was predicted(most confident). After the warmup they get the max without -1s???

    So we'll rename their variables to make sense:
    classwise_acc -> norm_class_frequency
    pseudo_counter: number of instances pseudo labeled for each class
    selected_label -> predicted label

    Overall pseudo code of OG:

    Once we actually have confidentally(0.95) predicted labels, begin calculating norm_class_freq with warmup version and non warmup

    Calculate consistency loss:
        -(use distribution alignment maybe)
        - Get the highest probability predicted for each instance in the batch
        - Check if greater than the dynamic threshold: (0.95 * norm_class_freq of most confident class / 2 - norm class frequency of most confident class)
        - Check if probability is greater than 0.95 (if it is it will be used when calculating norm_class_freq)
        - Calculate CE loss with strong logits and the pseudo-labels as the target, masked with the threshold
        - Encourages consistency between predictions and emphasizes confident predictions in the loss

    """

    p_cutoff = 0.95 # probability confidence cutoff
    pseudo_label = p_weak

    # probably do domain alignment here?
    # first implement the self-distillation method
    # Then adjust the hyperparameters specifically for FN(and maybe TN?)
    # The structured prior's self-distillation is used with their prior, then they use that to guide the distribution of the hard
    # Look at other self-supervised methods

    # Get top-k confident labels
    topk_probs, topk_idx = torch.topk(pseudo_label, K, dim=-1)
    topk_probs = topk_probs.to(device)
    topk_idx = topk_idx.to(device)
    norm_class_freq = norm_class_freq.to(device)

    # Compute a mask that decides which examples are reliable based on pseudo-label probabilities and class accuracy
    # High-confidence and high-accuracy examples contribute more to the loss, encouraging the model to focus on them
    # Compute dynamic threshold for each label and check which probs exceed the threshold
    dynamic_threshold = p_cutoff * (norm_class_freq[topk_idx] / (2. - norm_class_freq[topk_idx]))

    # mask is only size 15
    topk_mask = topk_probs.ge(dynamic_threshold).float()

    # Expand mask shape
    mask = torch.zeros_like(pseudo_label)  # Create zeros tensor of the same shape as pseudo_label
    # Advanced Indexing
    batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand_as(topk_idx)
    mask[batch_indices, topk_idx] = topk_mask
    confident_labels = topk_probs.ge(p_cutoff).long()

    loss = (BCELoss(p_strong, pseudo_label) * mask).sum(dim=1)

    return loss, confident_labels, topk_idx.long()


def calculate_classwise_accuracy(probs, true_labels, device, fixed_threshold=0.5):
    # Convert probabilities to binary predictions using the fixed threshold
    binary_preds = (probs > fixed_threshold).float().to(device)
    true_labels = true_labels.to(device)

    # Calculate the class-wise accuracy
    num_classes = probs.shape[1]
    classwise_acc = torch.zeros(num_classes)

    for i in range(num_classes):
        true_positive = torch.sum((binary_preds[:, i] == true_labels[:, i]) & (true_labels[:, i] == 1)).float()
        total_samples = torch.sum(true_labels[:, i]).float()

        if total_samples != 0:
            classwise_acc[i] = true_positive / total_samples

    return classwise_acc


def calculate_dynamic_threshold(classwise_acc, it, alpha=1, beta=0.5):
    # Assuming you want to use the same formula as in the FlexMatch code:
    # T = p_cutoff * (class_acc[max_idx] / (2. - class_acc[max_idx]))

    # Calculate the base threshold (p_cutoff) based on the current iteration (it)
    p_cutoff = alpha * (1 - math.exp(-beta * it))

    # Calculate the dynamic threshold (T) using classwise_acc
    T = p_cutoff * (classwise_acc / (2. - classwise_acc))

    return T


class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function,
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions.
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss.

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau # should test tau
        self.change_epoch = change_epoch # dont seemingly make a huge difference
        self.margin = margin
        self.gamma = gamma
        #self.reduction = reduction
        self.reduction = 'instance'

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch, entropy_weighting) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """


        logits = torch.where(targets == 1, logits - self.margin, logits)


        pred = logits
        if epoch >= self.change_epoch:
            targets = torch.where((logits > self.tau),
                torch.tensor(1).cuda(), targets)



        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt ** self.gamma

        # More logit bullshit, this takes the log of the sigmoid....which we are saying we dont need the sigmoid
        # so try just multiplying by log
        # This is sort of BCE
        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        label_level_loss= loss

        if entropy_weighting is not None:
            class_level = False
            if class_level:
                class_level_loss = loss * entropy_weighting
                instance_loss = class_level_loss.sum(dim=1)
            else:
                instance_loss = loss.sum(dim=1) * entropy_weighting

        else:
            instance_loss = loss.sum(dim=1)


        if self.reduction == 'mean':
            return loss.mean(), label_level_loss
        elif self.reduction == 'sum':
            return loss.sum(), label_level_loss
        elif self.reduction == 'instance':
            # HEYO return instance_loss.sum(), label_level_loss
            return instance_loss.sum(), label_level_loss
            #return loss.sum(dim=1), loss
        else:
            return loss

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter,
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss.

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPLC_resnet(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function,
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions.
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss.

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super(SPLC_resnet, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch, entropy_weighting=None):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if entropy_weighting is not None:
            instance_loss = loss.sum(dim=1) * entropy_weighting

        else:
            instance_loss = loss.sum(dim=1)

        return instance_loss.sum(), targets