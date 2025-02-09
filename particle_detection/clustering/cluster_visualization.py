import numpy as np
import matplotlib.pyplot as plt
from particle_detection.utils.visualization import (
    resize_and_convert,
    get_contours,
    filter_contours_by_area_and_shape,
    draw_and_display,
)

def process_and_visualize_clusters(sample_img, cluster_labels_grid, cluster_id=None, min_area=10, max_area=500, min_circularity=0.7, debug=False, save_path=None, no_filter=False, title=None):
    """
    Process and visualize clusters with options for filtering and focusing on specific clusters.

    Args:
        sample_img (torch.Tensor): Grayscale image tensor, shape [1, 1, H, W].
        cluster_labels_grid (np.ndarray): Cluster labels grid, shape [B, H, W].
        cluster_id (int): Specific cluster to focus on (None for all clusters).
        min_area (int): Minimum area for filtering particles.
        max_area (int): Maximum area for filtering particles.
        min_circularity (float): Minimum circularity for filtering particles.
        debug (bool): If True, print debug information.
        save_path (str): Optional. Path to save the final visualization.
        no_filter (bool): If True, skips filtering by size and circularity.
        title (str): Custom title for the visualization.
    """
    # Convert and resize
    image_np = sample_img[0, 0].cpu().numpy()
    original_h, original_w = image_np.shape
    image_colored = resize_and_convert(image_np, (original_w, original_h), to_rgb=True)
    cluster_labels_resized = resize_and_convert(cluster_labels_grid[0], (original_w, original_h))

    if cluster_id is not None:
        # Focus on a single cluster
        mask = (cluster_labels_resized == cluster_id).astype(np.uint8)
        contours = get_contours(mask)
        if debug:
            print(f"Cluster ID: {cluster_id}, Pixels: {np.sum(mask)}, Total Contours: {len(contours)}")
    else:
        # Process all clusters
        contours = []
        for unique_id in np.unique(cluster_labels_resized):
            mask = (cluster_labels_resized == unique_id).astype(np.uint8)
            contours.extend(get_contours(mask))
            if debug and unique_id != -1:
                print(f"Cluster ID: {unique_id}, Pixels: {np.sum(mask)}")

    if no_filter:
        valid_contours = contours
    else:
        # Filter contours by size and shape
        valid_contours = filter_contours_by_area_and_shape(contours, min_area, max_area, min_circularity)
        if debug:
            print(f"Valid Contours after Filtering: {len(valid_contours)}")

    # Draw and display results
    draw_and_display(
        image_colored, valid_contours, title=title if title else ("Filtered Particles" if cluster_id is None else f"Cluster {cluster_id}"),
        save_path=save_path
    )

def visualize_binary_clusters(cluster_labels_grid):
    """
    Display binary masks for each unique cluster in the cluster labels grid.

    Args:
        cluster_labels_grid (np.ndarray): Cluster labels grid, shape [B, H, W].

    Returns:
        None: Displays binary masks for each cluster.
    """
    # Print unique cluster labels
    unique_clusters = np.unique(cluster_labels_grid)
    print(f"Unique cluster labels: {unique_clusters}")

    # Iterate over unique clusters and display binary masks
    for cluster in unique_clusters:
        # Create a binary mask for the current cluster
        cluster_mask = (cluster_labels_grid == cluster).astype(np.uint8)

        if cluster_mask.ndim == 3:  # If 3D mask, sum along the batch dimension
            cluster_mask = np.sum(cluster_mask, axis=0)
        cluster_mask = (cluster_mask > 0).astype(np.uint8)

        # Display the binary mask
        plt.figure(figsize=(5, 5))
        plt.imshow(cluster_mask, cmap='gray')
        plt.title(f"Binary Mask for Cluster {cluster}")
        plt.axis("off")
        plt.show()

def compare_original_and_clusters(image_batch, cluster_labels_grid):
    """
    Visualize an image alongside its corresponding cluster assignments.

    Args:
        image_batch (torch.Tensor): Batch of images, shape [B, C, H, W].
        cluster_labels_grid (np.ndarray): Cluster labels, shape [B, H, W].

    Returns:
        None: Displays a side-by-side comparison of the original image and its clusters.
    """
    # Select the first image and its cluster labels
    single_image = image_batch[0]  # Shape: [C, H, W]
    single_cluster_labels = cluster_labels_grid[0]  # Shape: [H, W]

    # Visualize the image and clusters
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    image_np = single_image.cpu().numpy().transpose(1, 2, 0)  # Convert CHW to HWC
    if image_np.max() > 1.0:
        image_np = image_np / 255.0  # Normalize
    plt.imshow(image_np, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Cluster Visualization
    plt.subplot(1, 2, 2)
    plt.imshow(single_cluster_labels, cmap="viridis")
    plt.title("Cluster Visualization")
    plt.colorbar(label="Cluster")
    plt.axis("off")

    plt.tight_layout()
    plt.show()