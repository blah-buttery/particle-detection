import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import torch

def overlay_detection_results(img_tensor, true_path, pred_path, threshold=10, title="Detection Overlay", save_path=None):
    """Overlay detection results on an image by comparing ground truth and predicted coordinates.

    This function reads true and predicted particle coordinates from CSV files, matches them
    based on a distance threshold, and overlays true positives (TP), false positives (FP),
    and false negatives (FN) on the given image. The output can be displayed or saved as an image.

    Args:
        img_tensor (torch.Tensor): The input image tensor with shape (C, H, W) or (H, W).
        true_path (str): Path to the CSV file containing ground truth coordinates. Must contain "X" and "Y" columns.
        pred_path (str): Path to the CSV file containing predicted coordinates.
        threshold (float, optional): Distance threshold for matching predictions to ground truth. Defaults to 10.
        title (str, optional): Title of the plot. Defaults to "Detection Overlay".
        save_path (str, optional): If provided, saves the overlay plot to this path. If None, displays the plot.

    Returns:
        None
    """
    try:
        true_df = pd.read_csv(true_path, usecols=["X", "Y"])
        pred_df = pd.read_csv(pred_path)
    except Exception as e:
        print(f"Error reading CSV files ({true_path}, {pred_path}): {e}")
        return

    true_coords = list(zip(true_df["X"], true_df["Y"]))
    pred_coords = list(zip(pred_df.iloc[:, 0], pred_df.iloc[:, 1]))

    true_tree = cKDTree(true_coords)
    matched_true = set()
    TP_coords, FP_coords = [], []

    for pred in pred_coords:
        dist, idx = true_tree.query(pred, distance_upper_bound=threshold)
        if dist != np.inf and idx not in matched_true:
            TP_coords.append(pred)
            matched_true.add(idx)
        else:
            FP_coords.append(pred)

    FN_coords = [true_coords[i] for i in range(len(true_coords)) if i not in matched_true]

    img = img_tensor.squeeze().cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    if TP_coords:
        x_tp, y_tp = zip(*TP_coords)
        plt.scatter(x_tp, y_tp, facecolors="none", edgecolors="lime", label="TP", s=35, linewidths=0.5)
    if FP_coords:
        x_fp, y_fp = zip(*FP_coords)
        plt.scatter(x_fp, y_fp, facecolors="none", edgecolors="red", label="FP", s=35, linewidths=0.5)
    if FN_coords:
        x_fn, y_fn = zip(*FN_coords)
        plt.scatter(x_fn, y_fn, facecolors="none", edgecolors="orange", label="FN", s=35, linewidths=0.5)

    plt.legend()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def particle_count_accuracy(true_path, pred_path, threshold=10):
    """Calculate detection accuracy based on matching predicted and true coordinates.

    This function compares the predicted particle coordinates with the ground truth
    and computes the following metrics:
        - True count (ground truth total)
        - Predicted count (predictions total)
        - True positives (correct predictions within the threshold)
        - Detection accuracy as a percentage

    Args:
        true_path (str): Path to the CSV file containing ground truth coordinates. Must contain "X" and "Y" columns.
        pred_path (str): Path to the CSV file containing predicted coordinates.
        threshold (float, optional): Distance threshold within which a prediction is considered a true positive. Defaults to 10.

    Returns:
        dict: A dictionary containing:
            - "true_count" (int): Number of true labeled particles.
            - "predicted_count" (int): Number of predicted particles.
            - "true_positive" (int): Number of true positive matches.
            - "accuracy" (float): Detection accuracy as a percentage.
    """
    try:
        true_df = pd.read_csv(true_path, usecols=["X", "Y"])
        pred_df = pd.read_csv(pred_path)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        raise e

    true_count = len(true_df)
    predicted_count = len(pred_df)
    true_coords = list(zip(true_df["X"], true_df["Y"]))
    pred_coords = list(zip(pred_df.iloc[:, 0], pred_df.iloc[:, 1]))

    true_tree = cKDTree(true_coords)
    matched_true_indices = set()
    true_positive = 0

    for pred in pred_coords:
        dist, idx = true_tree.query(pred, distance_upper_bound=threshold)
        if dist != np.inf and idx not in matched_true_indices:
            true_positive += 1
            matched_true_indices.add(idx)

    accuracy = (true_positive / true_count * 100) if true_count > 0 else 0

    return {
        "true_count": true_count,
        "predicted_count": predicted_count,
        "true_positive": true_positive,
        "accuracy": accuracy
    }