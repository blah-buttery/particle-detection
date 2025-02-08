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
