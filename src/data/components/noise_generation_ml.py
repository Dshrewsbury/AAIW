import torch

"""
Injects instance-dependent noise into multi-label ground truth tensors.

Args:
    y_true (torch.Tensor): Ground truth label tensor of shape (num_samples, num_labels)
    classifier (nn.Module): Trained multi-label classifier
    input_tensors (torch.Tensor): Input tensors of shape (num_samples, input_shape)
    r (int): Percentage of samples to flip labels for each label

Returns:
    torch.Tensor: Noisy ground truth tensor of shape (num_samples, num_labels)
"""

# How to optimize prob_threshold? Should it be different per class? per instance?
def classification_based_noise(y_true, classifier, images, noise_percentage=30, prob_threshold=0.5):
    num_samples, num_labels = y_true.size()

    # Get the probabilities for each label
    predictions = []
    with torch.no_grad():
        batch_pred = torch.sigmoid(classifier(images))
        predictions.append(batch_pred)

    y_pred = torch.cat(predictions, dim=0)

    # Exclude the true labels from the y_pred matrix
    y_pred_no_true = y_pred.clone()
    y_pred_no_true[y_true.bool()] = -1

    # Get the maximum non-true-label probabilities
    label_noisy_prob = torch.max(y_pred_no_true, dim=1).values

    # Sort the instances by the maximum non-true-label probabilities
    _, sorted_indices = torch.sort(label_noisy_prob, descending=True)

    # Select the top instances to flip based on the noise rate
    num_instances_to_flip = int(num_samples * noise_percentage / 100)
    top_indices_to_flip = sorted_indices[:num_instances_to_flip]

    # Create a copy of the ground truth labels to introduce noise
    noisy_y_true = y_true.clone()

    # Flip the labels of the selected instances based on the probability threshold
    for index in top_indices_to_flip:
        flip_mask = (y_pred_no_true[index] > prob_threshold) & (y_true[index] != 1)
        noisy_y_true[index][flip_mask] = 1 - noisy_y_true[index][flip_mask]

    return noisy_y_true
