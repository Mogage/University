import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def mean_pixel_accuracy(predicted_images, ground_truth_images):
    correctly_predicted_pixels = np.sum(predicted_images - ground_truth_images < 0.001)
    total_pixels = np.prod(predicted_images.shape)
    return correctly_predicted_pixels / total_pixels


def intersection_over_union(predicted_images, ground_truth_images, class_id):
    intersection = np.sum(predicted_images[ground_truth_images - class_id < 0.001])
    union = np.sum(predicted_images) + np.sum(ground_truth_images) - intersection
    return intersection / union


def mean_intersection_over_union(predicted_images, ground_truth_images):
    iou_scores = []
    for class_id in np.unique(ground_truth_images):
        iou_scores.append(intersection_over_union(predicted_images, ground_truth_images, class_id))
    return np.mean(iou_scores)


def frequency_weighted_intersection_over_union(predicted_images, ground_truth_images):
    fw_iou_scores = []
    for class_id in np.unique(ground_truth_images):
        class_frequencies = np.sum(ground_truth_images - class_id < 0.001)
        iou_score = intersection_over_union(predicted_images, ground_truth_images, class_id)
        fw_iou_scores.append(iou_score * class_frequencies)
    return np.sum(fw_iou_scores) / np.sum(ground_truth_images)


def metrics_0(predicted_images, ground_truth_images):
    num_classes = 3
    score = 0.0
    n = np.zeros((num_classes, num_classes))
    t = np.zeros(num_classes)

    for class_id_i in range(num_classes):
        for class_id_j in range(num_classes):
            n[class_id_i][class_id_j] = np.sum(
                predicted_images[ground_truth_images - class_id_i < 0.001] - class_id_j < 0.001)

    for class_id in range(num_classes):
        t[class_id] = np.sum(ground_truth_images - class_id < 0.001)

    for class_id in range(num_classes):
        score += n[class_id][class_id] / t[class_id]

    return score / num_classes


def metrics_1(predicted_images, ground_truth_images):
    num_classes = 3
    score = 0.0
    n = np.zeros((num_classes, num_classes))
    t = np.zeros(num_classes)

    for class_id_i in range(num_classes):
        for class_id_j in range(num_classes):
            n[class_id_i][class_id_j] = np.sum(
                predicted_images[ground_truth_images - class_id_i < 0.001] - class_id_j < 0.001)

    for class_id in range(num_classes):
        t[class_id] = np.sum(ground_truth_images - class_id < 0.001)

    for class_id_i in range(num_classes):
        second_sum = 0
        for class_id_j in range(num_classes):
            second_sum += n[class_id_j][class_id_i]
        # union = t[class_id_i] + np.sum(predicted_images - class_id_i < 0.001) - n[class_id_i][class_id_i]
        union = t[class_id_i] + second_sum - n[class_id_i][class_id_i]
        score += n[class_id_i][class_id_i] / union

    return score / num_classes


def metrics_2(predicted_images, ground_truth_images):
    num_classes = 3
    score = 0.0
    t = np.zeros(num_classes)

    # n = np.zeros((num_classes, num_classes))

    #
    # for class_id_i in range(num_classes):
    #     for class_id_j in range(num_classes):
    #         n[class_id_i][class_id_j] = np.sum(
    #             predicted_images[ground_truth_images - class_id_i < 0.001] - class_id_j < 0.001)

    n = metrics.confusion_matrix(ground_truth_images.flatten(), predicted_images.flatten())

    for class_id in range(num_classes):
        # t[class_id] = np.sum(ground_truth_images - class_id < 0.001)
        t[class_id] = np.sum(n[class_id])

    for class_id_i in range(num_classes):
        second_sum = 0
        for class_id_j in range(num_classes):
            second_sum += n[class_id_j][class_id_i]
        # union = t[class_id_i] + np.sum(predicted_images - class_id_i < 0.001) - n[class_id_i][class_id_i]
        union = t[class_id_i] + second_sum - n[class_id_i][class_id_i]
        score += t[class_id_i] * n[class_id_i][class_id_i] / union
        # class_frequencies = np.sum(ground_truth_images - class_id_i < 0.001)
        score *= (1 / t[class_id_i])

    return score / num_classes


def evaluate_model(model, data_loader, device, experiment):
    model.to(device)
    model.eval()

    mean_pixel_accuracy_score = 0.0
    intersection_over_union_score = 0.0
    frequency_weighted_intersection_over_union_score = 0.0
    score_0 = 0.0
    score_1 = 0.0
    score_2 = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # outputs = F.interpolate(outputs, size=(labels.size(-2), labels.size(-1)), mode='bilinear', align_corners=True)
        predicted_images = outputs.detach().cpu().numpy()
        ground_truth_images = labels.cpu().numpy()

        # fig, axs = plt.subplots(1, 2)
        # print(predicted_images.squeeze().shape)
        # axs[0].imshow(predicted_images[0].squeeze().transpose(1, 2, 0))
        # axs[1].imshow(ground_truth_images[0].squeeze().transpose(1, 2, 0))
        # plt.show()

        # predicted_images = np.argmax(predicted_images, axis=1)
        # ground_truth_images = np.argmax(ground_truth_images, axis=1)

        score_0 += metrics_0(predicted_images, ground_truth_images)
        score_1 += metrics_1(predicted_images, ground_truth_images)
        score_2 += metrics_2(predicted_images, ground_truth_images)

        mean_pixel_accuracy_score += mean_pixel_accuracy(predicted_images, ground_truth_images)
        intersection_over_union_score += mean_intersection_over_union(predicted_images, ground_truth_images)
        frequency_weighted_intersection_over_union_score += frequency_weighted_intersection_over_union(predicted_images,
                                                                                                       ground_truth_images)

    mean_pixel_accuracy_score /= len(data_loader)
    intersection_over_union_score /= len(data_loader)
    frequency_weighted_intersection_over_union_score /= len(data_loader)
    score_0 /= len(data_loader)
    score_1 /= len(data_loader)
    score_2 /= len(data_loader)

    print(score_0)
    print(score_1)
    print(score_2)

    # experiment.log(
    #     {
    #         'mean pixel accuracy': mean_pixel_accuracy_score,
    #         'mean intersection over union': intersection_over_union_score,
    #         'frequency weighted intersection over union': frequency_weighted_intersection_over_union_score
    #     }
    # )

    return mean_pixel_accuracy_score, intersection_over_union_score, frequency_weighted_intersection_over_union_score
