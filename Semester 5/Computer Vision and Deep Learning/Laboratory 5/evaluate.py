import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import wandb


def compute_n_t(predicted_images, ground_truth_images, num_classes=3):
    t = np.zeros(num_classes)

    n = metrics.confusion_matrix(ground_truth_images.flatten(), predicted_images.flatten())

    for class_id in range(num_classes):
        t[class_id] = np.sum(n[class_id])

    return n, t


def mean_pixel_accuracy(predicted_images, ground_truth_images, num_classes=3):
    n, t = compute_n_t(predicted_images, ground_truth_images)
    score = 0.0

    for class_id in range(num_classes):
        score += n[class_id][class_id] / t[class_id]

    return score / num_classes


def mean_intersection_over_union(predicted_images, ground_truth_images, num_classes=3):
    n, t = compute_n_t(predicted_images, ground_truth_images)
    score = 0.0

    for class_id_i in range(num_classes):
        second_sum = 0
        for class_id_j in range(num_classes):
            second_sum += n[class_id_j][class_id_i]
        union = t[class_id_i] + second_sum - n[class_id_i][class_id_i]
        score += n[class_id_i][class_id_i] / union

    return score / num_classes


def frequency_weighted_intersection_over_union(predicted_images, ground_truth_images, num_classes=3):
    n, t = compute_n_t(predicted_images, ground_truth_images)
    score = 0.0

    for class_id_i in range(num_classes):
        second_sum = 0
        for class_id_j in range(num_classes):
            second_sum += n[class_id_j][class_id_i]
        union = t[class_id_i] + second_sum - n[class_id_i][class_id_i]
        score += t[class_id_i] * n[class_id_i][class_id_i] / union

    score *= np.power(np.sum(t), -1)
    return score


def evaluate_model(model, data_loader, device, experiment):
    model.to(device)
    model.eval()

    mean_pixel_accuracy_score = 0.0
    intersection_over_union_score = 0.0
    frequency_weighted_intersection_over_union_score = 0.0

    wandb_table = wandb.Table(columns=["predicted",
                                       "ground_truth",
                                       "mean_pixel_accuracy_score",
                                       "intersection_over_union_score",
                                       "frequency_weighted_intersection_over_union_score"])

    last_output = None
    last_label = None

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # outputs = F.interpolate(outputs, size=(labels.size(-2), labels.size(-1)), mode='bilinear', align_corners=True)
        outputs = outputs.detach().cpu().numpy()
        labels = labels.cpu().numpy()

        predicted_images = np.argmax(outputs, axis=1)
        ground_truth_images = np.argmax(labels, axis=1)

        mean_pixel_accuracy_score += mean_pixel_accuracy(predicted_images, ground_truth_images)
        intersection_over_union_score += mean_intersection_over_union(predicted_images, ground_truth_images)
        frequency_weighted_intersection_over_union_score += frequency_weighted_intersection_over_union(predicted_images,
                                                                                                       ground_truth_images)

        last_output = outputs[0].transpose(1, 2, 0)
        last_label = labels[0].transpose(1, 2, 0)

    mean_pixel_accuracy_score /= len(data_loader)
    intersection_over_union_score /= len(data_loader)
    frequency_weighted_intersection_over_union_score /= len(data_loader)

    wandb_table.add_data(
        wandb.Image(last_output),
        wandb.Image(last_label),
        mean_pixel_accuracy_score,
        intersection_over_union_score,
        frequency_weighted_intersection_over_union_score
    )

    experiment.log(
        {
            'mean pixel accuracy': mean_pixel_accuracy_score,
            'mean intersection over union': intersection_over_union_score,
            'frequency weighted intersection over union': frequency_weighted_intersection_over_union_score,
            'metrics': wandb_table
        }
    )

    return mean_pixel_accuracy_score, intersection_over_union_score, frequency_weighted_intersection_over_union_score
