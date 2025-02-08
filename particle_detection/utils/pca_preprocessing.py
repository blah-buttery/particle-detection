import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

def plot_explained_variance(pca, title="Explained Variance by Number of Components"):
  """
  Plot the cumulative explained variance for a fitted PCA model.

  Args:
    pca (PCA): Fitted PCA model from sklearn.decomposition.PCA.
    title (str): Title of the plot. Default is "Explained Variance by Number of Components".

  Returns:
    None: Displays the explained variance plot.
  """
  explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
  plt.xlabel("Number of Principal Components")
  plt.ylabel("Cumulative Explained Variance")
  plt.title(title)
  plt.grid()
  plt.show()

# Example usage (can be removed if not needed in the final file)
if __name__ == "__main__":
  # Simulate some data
  np.random.seed(42)
  data = np.random.rand(100, 50)  # 100 samples, 50 features

  # Apply PCA
  reduced_data, pca_model = apply_pca(data, explained_variance_ratio=0.95, debug=True)

  # Plot explained variance
  plot_explained_variance(pca_model)

