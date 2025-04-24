import argparse
import json
import os
import torch
from particle_detection.vae.vae_loader import load_vae
from particle_detection.utils.patch_utils import extract_latent_representations
from particle_detection.utils.clustering import apply_pca, cluster_view, visualize_clusters, plot_k_distance, label_clusters
from particle_detection.utils.generate_report import create_pdf_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from particle_detection.data.data_pipeline import create_dataloaders

def run_eval(model_path, image_dir, csv_dir, save_dir, batch_size=8, eps=8.0, min_samples=4, pca_components=50, threshold=10):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vae = load_vae(model_path, device=device)
    _, test_loader = create_dataloaders(data_dir=image_dir, image_size=(2048, 2048), batch_size=batch_size)

    text_results = []
    print(f"Saving results to {save_dir}")

    try:
        test_batch = next(iter(test_loader))
    except StopIteration:
        print("Test dataloader is empty.")
        return
      
    latent = extract_latent_representations(test_loader, vae, patch_size=16, device=device)
  
    for i, img_tensor in enumerate(test_batch):
        print(f"\nProcessing Test Image {i}")
      
        latent_normalized = StandardScaler().fit_transform(latent[i].cpu().numpy())
        pca, _ = apply_pca(latent_normalized, n_components=pca_components)

        #plot_k_distance(pca, k=5, save_path=os.path.join(save_dir, f"k_distance_{i}.png"))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(pca)

        cluster_view(cluster_labels, pca, save_path=os.path.join(save_dir, f"cluster_view_{i}.png"))
      
        visualize_clusters(
            img_tensor.cpu(), cluster_labels, patch_size=16,
            save_path=os.path.join(save_dir, f"visualize_clusters_{i}.png")
        )
      
        num_particles = label_clusters(
            img_tensor, cluster_labels, patch_size=16, cluster_id=-1,
            title=f"Test Image {i} Clusters",
            save_path=os.path.join(save_dir, f"label_clusters_{i}.png")
        )

        text_results.append((i, {"num_particles": num_particles}))

    create_pdf_report(save_dir, text_results)
    print("Evaluation complete and report generated!")

def get_args():
    config_parser = argparse.ArgumentParser(description="Nanoparticle Detection Eval", add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to JSON config file", default=None)
    args_config, remaining_args = config_parser.parse_known_args()

    config_defaults = {}
    if args_config.config:
        with open(args_config.config, "r") as f:
            config_defaults = json.load(f)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--model_path", type=str, help="Path to the saved VAE model")
    parser.add_argument("--image_dir", type=str, help="Directory containing test images")
    parser.add_argument("--csv_dir", type=str, help="Directory containing CSV labels")
    parser.add_argument("--save_dir", type=str, help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--min_samples", type=int, default=4)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=10)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_args)
    return args

if __name__ == "__main__":
    args = get_args()
    print("Effective Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    args_dict = vars(args)
    args_dict.pop("config", None)
    run_eval(**args_dict)