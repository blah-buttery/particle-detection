import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import cv2
from sklearn.manifold import TSNE

def apply_pca(data, n_components=30, explained_variance_ratio=0.95, debug=False):
  """
  Apply PCA on the given data to reduce dimensionality.

  Args:
      data (np.ndarray): Input data of shape (n_samples, n_features).
      n_components (int): Number of components for PCA. Default is 30.
      explained_variance_ratio (float): If provided, overrides `n_components`
          to retain this fraction of variance. Default is 0.95.
      debug (bool): If True, prints PCA debug information.

  Returns:
      reduced_data (np.ndarray): Transformed data of shape (n_samples, n_components).
      pca_model (PCA): Trained PCA model (can be reused for inverse transforms, etc.).
  """
  pca = PCA(n_components=n_components)
   
  if explained_variance_ratio:
    pca = PCA(n_components=explained_variance_ratio)
    
  reduced_data = pca.fit_transform(data)
    
  if debug:
    print(f"PCA Components: {pca.n_components_}")
    print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

  return reduced_data, pca

def plot_k_distance(data, k=5, save_path=None):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances[:, k - 1], axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.title("K-distance Graph")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th Nearest Neighbor Distance")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def cluster_view(cluster_labels, latent_space, save_path=None, perplexity=30, random_state=42):
    """
    Visualizes cluster labels on t-SNE reduced latent space.

    Args:
        cluster_labels (ndarray): Cluster labels from DBSCAN (or other clustering).
        latent_space (ndarray): Latent space vectors (before reduction).
        save_path (str, optional): If provided, saves the plot to this path.
        perplexity (int): t-SNE perplexity parameter.
        random_state (int): Random seed for reproducibility.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(latent_space)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, s=5)
    plt.title("t-SNE of Latent Space with Cluster Labels")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_clusters(img, cluster_labels, patch_size=16, stride=8, save_path=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    height, width = img.shape[1], img.shape[2]
    clustered_image = np.zeros((height, width))
    patch_idx = 0
    weight_matrix = np.zeros((height, width))
    
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            clustered_image[i:i + patch_size, j:j + patch_size] += cluster_labels[patch_idx]
            weight_matrix[i:i + patch_size, j:j + patch_size] += 1
            patch_idx += 1

    clustered_image /= np.maximum(weight_matrix, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(clustered_image, cmap="viridis")
    plt.colorbar()
    plt.title("Nanoparticle Clusters with Stride")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def normalize_image(image):
    """Normalize an image tensor/array to 0–255 uint8 scale."""
    image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255  # Added small epsilon for safety
    return image.astype(np.uint8)

def label_clusters(sample_img, cluster_labels, patch_size=16, stride=8, cluster_id=None, 
                   title="Cluster Contours", save_path=None):
    """
    Overlay cluster contours on an image and optionally save the result.

    Args:
        sample_img (Tensor): The original image tensor (C, H, W).
        cluster_labels (ndarray): Cluster labels for each patch.
        patch_size (int): Patch size used during clustering.
        stride (int): Stride used during clustering.
        cluster_id (int, optional): If specified, only highlight this cluster.
        title (str): Title for the plot.
        save_path (str, optional): If given, saves the plot instead of showing.

    Returns:
        int: Number of detected contours (particles).
    """
    img_array = sample_img.cpu().numpy().squeeze()
    img_h, img_w = img_array.shape[:2]
    image_np = normalize_image(img_array)

    cluster_grid = np.zeros((img_h, img_w))
    weight_matrix = np.zeros((img_h, img_w))

    patch_idx = 0
    for i in range(0, img_h - patch_size + 1, stride):
        for j in range(0, img_w - patch_size + 1, stride):
            cluster_grid[i:i + patch_size, j:j + patch_size] += cluster_labels[patch_idx]
            weight_matrix[i:i + patch_size, j:j + patch_size] += 1
            patch_idx += 1

    cluster_grid /= np.maximum(weight_matrix, 1)
    resized_mask = cv2.resize(cluster_grid, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # Mask selection logic
    if cluster_id is not None:
        mask = (resized_mask == cluster_id).astype(np.uint8)
    else:
        mask = (resized_mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_particles = len(contours)

    overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(overlay)
    plt.title(f"{title} | Particles Found: {num_particles}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return num_particles