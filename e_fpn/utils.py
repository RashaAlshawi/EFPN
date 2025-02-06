import numpy as np
import keras.backend as K
from sklearn.metrics import f1_score, matthews_corrcoef


def calculate_fwiou(predicted_mask, ground_truth_mask, class_frequencies):
    predicted_flat = K.flatten(predicted_mask)
    ground_truth_flat = K.flatten(ground_truth_mask)

    unique_classes = np.unique(ground_truth_flat)

    intersection_sum = 0
    union_sum = 0

    for class_val in unique_classes:
        class_mask = K.equal(ground_truth_flat, class_val)
        intersection = K.sum(K.cast(predicted_flat[class_mask] == class_val, K.floatx()))
        union = K.sum(K.cast((predicted_flat == class_val) | class_mask, K.floatx()))
        frequency = class_frequencies[class_val]

        intersection_sum += frequency * intersection
        union_sum += frequency * union

    fwiou = intersection_sum / union_sum
    return fwiou

def calculate_pixel_acc(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."
    total_pixels = np.prod(y_true.shape)
    correct_pixels = np.sum(y_true == y_pred)
    pixel_acc = correct_pixels / total_pixels
    return pixel_acc


def calculate_balanced_accuracy(y_true, y_pred):
    unique_classes = np.unique(y_true)
    num_classes = len(unique_classes)
    class_accuracies = []

    for cls in unique_classes:
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)
        class_accuracy = np.sum(true_mask & pred_mask) / np.sum(true_mask)
        class_accuracies.append(class_accuracy)

    balanced_acc = np.mean(class_accuracies)
    return balanced_acc


def calculate_iou(y_test, y_predict):
    # Flatten the input arrays
    y_true_flat = y_test.flatten()
    y_pred_flat = y_predict.flatten()

    # Get unique class values from both true and predicted arrays
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    
    # Initialize dictionary to store IoU scores for each class
    iou_scores = {}

    # Calculate IoU for each unique class
    for class_value in unique_classes:
        intersection = np.sum((y_true_flat == class_value) & (y_pred_flat == class_value))
        union = np.sum((y_true_flat == class_value) | (y_pred_flat == class_value))
        iou_scores[class_value] = intersection / union

    # Print IoU for each class
    for class_value, iou_score in iou_scores.items():
        print(f"IoU for class {class_value}: {iou_score}")

    # Calculate and print average IoU for all classes (including class 0)
    average_iou_all = np.mean(list(iou_scores.values()))
    print(f"Average IoU (including class 0): {average_iou_all}")

    # Calculate and print average IoU excluding class 0
    iou_scores_without_class_0 = {k: v for k, v in iou_scores.items() if k != 0}
    average_iou_without_class_0 = np.mean(list(iou_scores_without_class_0.values()))
    print(f"Average IoU (excluding class 0): {average_iou_without_class_0}")

    return average_iou_all, average_iou_without_class_0, iou_scores



def calculate_f1_score(y_test, y_predict):
  
    # Flatten the input arrays
    y_true_flat = y_test.flatten()
    y_pred_flat = y_predict.flatten()

    # Get unique class values from both true and predicted arrays
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    
    # Initialize dictionary to store F1 scores for each class
    f1_scores = {}

    # Calculate F1 score for each unique class
    for class_value in unique_classes:
        binary_y_true = (y_true_flat == class_value)
        binary_y_pred = (y_pred_flat == class_value)
        f1_scores[class_value] = f1_score(binary_y_true, binary_y_pred)

    # Print F1 score for each class
    for class_value, f1_score_value in f1_scores.items():
        print(f"F1 Score for class {class_value}: {f1_score_value}")

    # Calculate and print average F1 score for all classes (including class 0)
    average_F1_all = np.mean(list(f1_scores.values()))
    print(f"Average F1 Score (including class 0): {average_F1_all}")

    # Calculate and print average F1 score excluding class 0
    f1_scores_without_class_0 = {k: v for k, v in f1_scores.items() if k != 0}
    average_F1_without_class_0 = np.mean(list(f1_scores_without_class_0.values()))
    print(f"Average F1 Score (excluding class 0): {average_F1_without_class_0}")

    return average_F1_all, average_F1_without_class_0, f1_scores



def map_to_rgb(image, color_map):
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rgb_image[i, j] = color_map[image[i, j]]
    return rgb_image
