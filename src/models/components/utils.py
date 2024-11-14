import torch
import random
import numpy as np
import math
from src.models.components import metrics
from PIL import ImageDraw


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    gcn = []
    gcn_no_decay = []
    prefix = "module." if torch.cuda.device_count() > 1 else ""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.startswith(f"{prefix}gc"):
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gcn_no_decay.append(param)
            else:
                gcn.append(param)
            #assert("gcn" in cfg.model_name)
        elif len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }, {
        'params': gcn_no_decay,
        'weight_decay': 0.
    }, {
        'params': gcn,
        'weight_decay': weight_decay
    }]


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


def compute_metrics(y_pred, y_true):
    '''
    Given predictions and labels, compute a few metrics.
    '''

    num_examples, num_classes = np.shape(y_true)

    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32)  # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))

    results['map'] = 100.0 * float(np.mean(average_precision_list))
    results['ap'] = 100.0 * np.array(average_precision_list)

    return results


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_jaccard_index(preds, labels):
    # Ensure valid_labels is a boolean mask of same shape as labels
    valid_labels = (labels != -1)

    # Apply the mask
    valid_preds = preds[valid_labels]
    valid_labels = labels[valid_labels]

    # Compute the intersection and union for each instance
    intersection = torch.sum(valid_preds * valid_labels, dim=-1)
    union = torch.sum(valid_preds, dim=-1) + torch.sum(valid_labels, dim=-1) - intersection

    # Compute Jaccard index for each instance
    jaccard = intersection / (union + 1e-6)  # add small constant to prevent division by zero

    return jaccard

def compute_hardness(s_logits, t_logits, partial_labels, batch_index):

    s_probs = torch.sigmoid(s_logits)
    t_probs = torch.sigmoid(t_logits)

    # Compute Jaccard Index with student and teacher
    s_preds = (s_probs > 0.5).float()
    t_preds = (t_probs > 0.5).float()
    jaccard = compute_jaccard_index(s_preds, t_preds)

    # Compute Average Confidence
    # confidence in positive predicted labels-> If the model is very confident in its labels, it will reduce
    # the jaccard index less, and thus result in a lower hardness. If it thinks it positive but isnt confident the hardness score will be higher
    # the lowest the average confidence could be is 0.5
    confidence_default = torch.tensor([0.01]).to(s_logits.device)
    s_confident_probs = s_probs[s_preds == 1]
    s_confidence = torch.mean(s_confident_probs, dim=-1) if s_confident_probs.numel() > 0 else confidence_default
    t_confident_probs = t_probs[t_preds == 1]
    t_confidence = torch.mean(t_confident_probs, dim=-1) if t_confident_probs.numel() > 0 else confidence_default
    average_confidence = (s_confidence + t_confidence) / 2


    # Unlabeled hardness measure is based off of Jaccard Index and average confidence of student/teacher
    unlabeled_hardness = 1 - jaccard * average_confidence

    # Labeled hardness measure based on the Jaccard Index between the student predictions and known labels
    # But only compare for labels that actually exist(some will be -1 for known labels and shouldnt be a part of the calculation)
    valid_labels = (partial_labels != -1)

    # Expand dimensions for valid_labels
    valid_labels_exp = valid_labels.unsqueeze(1)

    # Mask out the invalid labels
    s_preds_valid = s_preds * valid_labels_exp.float()
    partial_labels_valid = partial_labels * valid_labels_exp.float()

    # Apply filter along last dimension
    jaccard_labeled = compute_jaccard_index(s_preds_valid, partial_labels_valid)

    labeled_hardness = 1 - jaccard_labeled

    # Create Weighted hardness measure based off of unlabeled hardness and labeled hardness
    # the weight depends on how many labels were missing/-1 from partial_labels, if 60% of labels ar emissing the unlabeled hardness should have a weight of 0.6
    missing_label_ratio = torch.mean((partial_labels == -1).float(), dim=1)
    hardness = missing_label_ratio * unlabeled_hardness + (1 - missing_label_ratio) * labeled_hardness
    is_hard = (hardness > 0.5)
    # This is a hardness for the batch, should the score be per instance?

    return hardness, is_hard

def compute_hardness_loss(s_logits, t_logits, partial_labels, batch_index):

    # Only sum up losses for the missing labels?
    # then we'd still want to use AN to calculate the loss based off of observations
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    s_loss = criterion(s_logits, partial_labels)
    t_loss = criterion(t_logits, partial_labels)

    hardness = torch.abs(s_loss - t_loss)

    hardness = hardness.sum(dim=-1)



    # Labeled hardness measure based on the Jaccard Index between the student predictions and known labels
    # But only compare for labels that actually exist(some will be -1 for known labels and shouldnt be a part of the calculation)
    #valid_labels = (partial_labels != -1)

    # Expand dimensions for valid_labels
    #valid_labels_exp = valid_labels.unsqueeze(1)

    # Mask out the invalid labels
    #s_preds_valid = s_preds * valid_labels_exp.float()
    #partial_labels_valid = partial_labels * valid_labels_exp.float()

    # Apply filter along last dimension
    #jaccard_labeled = compute_jaccard_index(s_preds_valid, partial_labels_valid)

    #labeled_hardness = 1 - jaccard_labeled

    # Create Weighted hardness measure based off of unlabeled hardness and labeled hardness
    # the weight depends on how many labels were missing/-1 from partial_labels, if 60% of labels ar emissing the unlabeled hardness should have a weight of 0.6
    #missing_label_ratio = torch.mean((partial_labels == -1).float(), dim=1)
    #hardness = missing_label_ratio * unlabeled_hardness + (1 - missing_label_ratio) * labeled_hardness
    #is_hard = (hardness > 0.5)
    # This is a hardness for the batch, should the score be per instance?

    return hardness


def compute_misclassification_rate(predictions, labels):
    predicted_probs = torch.sigmoid(predictions)
    predicted_classes = (predicted_probs > 0.5).float()
    misclassified = predicted_classes != labels
    return misclassified.float().mean(dim=1)


# Updates the moving average for confidence scores
def update_moving_average(confidence_sums, confidence_weights, s_probs, partial_labels):
    # s_probs and partial_labels are tensors of shape [batch_size, n]
    # where n is the number of classes
    # confidence_sums and confidence_weights are tensors of shape [n]

    # Update the sums and weights only for the confident labels
    mask = (s_probs > 0.5) & partial_labels
    confidence_sums += torch.sum(s_probs * mask, dim=0)
    confidence_weights += torch.sum(mask, dim=0)

    return confidence_sums, confidence_weights

def build_optimizer(args, model):
    params_name = None

    # Learning rate decay
    params, param_group_names = param_groups_lrd(model, args.fix_layer, args.weight_decay,
                                                 layer_decay=args.layer_decay
                                                 )
    params_name = []
    for k, v in param_group_names.items():
        params_name += v["params"]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for name, param in model.named_parameters():
        if name not in params_name:
            param.requires_grad = False

    return optimizer

def param_groups_lrd(model, fix_layer, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    elif hasattr(model, "transformer"):
        num_layers = model.transformer.layers + 1
    else:
        num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        # import pdb; pdb.set_trace()
        if hasattr(model, "blocks"):
            layer_id = get_layer_id_for_vit(n, num_layers)
        else:
            layer_id = get_layer_id_for_clip(n, num_layers)

        if layer_id > fix_layer:

            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

    return list(param_groups.values()), param_group_names

def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def get_layer_id_for_clip(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """

    if name in ['cls_token', 'pos_embed', "class_embedding"]:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('conv1'):
        return 0
    elif name.startswith('ln_pre'):
        return 0
    elif name.startswith('positional_embedding'):
        return 0
    elif name.startswith('transformer.resblocks'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers

def get_layer_id_for_resnet(param_name):
    if 'layer1' in param_name:
        print('FINNA')
        return 1
    elif 'layer2' in param_name:
        return 2
    elif 'layer3' in param_name:
        return 3
    elif 'layer4' in param_name:
        return 4
    else:
        return 0  # for parameters not belonging to any specific layer, adjust as needed



def update_teacher_model(student_model, teacher_model, warmup_epochs, ema_decay_base, loader_len, i_iter):
    with torch.no_grad():

        # For warmup
        # ema_decay = min(
        #     1 - 1 / (i_iter - loader_len * warmup_epochs + 1),
        #     ema_decay_base,
        # )

        ema_decay = min(
            1 - 1 / (i_iter - loader_len),
            ema_decay_base,
        )

        # update parameters of the teacher model
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = teacher_param.data * ema_decay + student_param.data * (1 - ema_decay)
        # update buffers of the teacher model
        for student_buffer, teacher_buffer in zip(student_model.buffers(), teacher_model.buffers()):
            teacher_buffer.data = teacher_buffer.data * ema_decay + student_buffer.data * (1 - ema_decay)

# still calculating entropy at a class level independentally
# 1e-10 was what i originally used, their paper uses 1e-5
def calculate_class_entropy(y_pred):
    # Calculate entropy for each instance based on predicted probabilities
    #entropy = -torch.sum(y_pred * torch.log2(y_pred + 1e-5), dim=1)  # Add a small constant to avoid log(0)
    # Calculate entropy for each instance based on predicted probabilities
    entropy = -y_pred * torch.log2(y_pred + 1e-5) - (1 - y_pred) * torch.log2(1 - y_pred + 1e-5)
    #entropy = torch.sum(entropy, dim=1)  # Summing across the label dimension
    return entropy

def calculate_instance_entropy(y_pred):
    # Calculate entropy for each instance based on predicted probabilities
    entropy = -torch.sum(y_pred * torch.log2(y_pred + 1e-5), dim=1)  # Add a small constant to avoid log(0)
    # Calculate entropy for each instance based on predicted probabilities
    #entropy = -y_pred * torch.log2(y_pred + 1e-5) - (1 - y_pred) * torch.log2(1 - y_pred + 1e-5)
    #entropy = torch.sum(entropy, dim=1)  # Summing across the label dimension
    return entropy
