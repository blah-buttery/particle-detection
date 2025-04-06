import os
import json
import datetime
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from particle_detection.data.data_pipeline import create_dataloaders
from particle_detection.vae.model import create_vae
from particle_detection.utils.ddp_utils import setup_ddp, cleanup_ddp
from particle_detection.utils.model_utils import save_model, save_metrics_and_plots

def track_metrics(metrics, epoch_values):
    """
    Updates the metrics dictionary with the provided epoch values, with no console printing.

    Args:
        metrics (dict): Dictionary containing lists to track metrics.
        epoch_values (dict): A dictionary containing metric values for the current epoch, e.g.:
                             {"total_losses": 300.0, "recon_losses": 200.0, "kl_losses": 50.0, ...}.
    """
    for key, value in epoch_values.items():
        metrics.setdefault(key, [])
        metrics[key].append(value)

def train(rank, world_size, is_ddp, track_metrics_flag, num_epochs, batch_size, learning_rate,
          data_dir, image_size=(128, 128), beta=1.0, contrastive_margin=2.0,
          contrastive_scale=1.0, latent_dim=128):
    """
    Train the Variational Autoencoder (VAE) model.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Print device info only on rank 0
    if rank == 0:
        print(f"[Rank {rank}] Using device: {device}")

    train_loader, _ = create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        is_ddp=is_ddp
    )

    vae = create_vae(latent_dim=latent_dim)
    vae.to(device)

    if is_ddp:
        vae = DDP(vae, device_ids=[rank])

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    def vae_loss(recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + (beta * kl_div), recon_loss, kl_div

    def contrastive_loss(z1, z2, y, margin=1.0):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        euclidean_distance = F.pairwise_distance(z1, z2)
        loss = (1 - y) * euclidean_distance.pow(2) + \
               y * torch.clamp(margin - euclidean_distance, min=0.0).pow(2)
        return loss.mean()

    # Load pre-saved contrastive patches and labels
    z1 = torch.load('saved_models/patch1.pt')
    z2 = torch.load('saved_models/patch2.pt')
    labels = torch.load('saved_models/labels.pt')

    # Initialize metrics dictionary
    metrics = {
        "total_losses": [],
        "recon_losses": [],
        "kl_losses": [],
        "contrastive_losses": []
    }

    for epoch in range(num_epochs):
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_total_loss = 0.0

        # Shuffle contrastive patches
        perm = torch.randperm(len(z1))
        z1 = z1[perm]
        z2 = z2[perm]
        labels = labels[perm]

        # Only rank 0 displays the progress bar
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        else:
            progress_bar = train_loader

        contrastive_i = 0
        for i, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # VAE forward pass
            reconstructed, mu, logvar = vae(batch)
            total_loss, recon_loss, kl_loss = vae_loss(reconstructed, batch, mu, logvar, beta=beta)

            # Contrastive part
            if contrastive_i < len(z1):
                z1_patch = z1[contrastive_i].to(device).unsqueeze(0)
                z2_patch = z2[contrastive_i].to(device).unsqueeze(0)
                label = labels[contrastive_i].to(device).unsqueeze(0)

                vae.eval()
                with torch.no_grad():
                    z1_patch = F.interpolate(z1_patch, size=(128, 128), mode="bilinear", align_corners=False)
                    z2_patch = F.interpolate(z2_patch, size=(128, 128), mode="bilinear", align_corners=False)
                    if is_ddp:
                        z1_encoded = vae.module.encode(z1_patch)
                        z2_encoded = vae.module.encode(z2_patch)
                    else:
                        z1_encoded = vae.encode(z1_patch)
                        z2_encoded = vae.encode(z2_patch)
                vae.train()

                c_loss = contrastive_loss(z1_encoded, z2_encoded, label, margin=contrastive_margin)
                final_loss = total_loss + contrastive_scale * c_loss
                epoch_contrastive_loss += c_loss.item()
            else:
                final_loss = total_loss

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_total_loss += final_loss.item()

            if rank == 0:
                progress_bar.set_postfix(loss=f"{final_loss.item():.4f}")

            contrastive_i += 1

        # End of epoch: compute averages
        avg_loss = epoch_total_loss / len(train_loader)
        avg_kl = epoch_kl_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_contrastive = epoch_contrastive_loss / len(train_loader)

        # Update metrics only on rank 0
        if track_metrics_flag and rank == 0:
            epoch_values = {
                "total_losses": avg_loss,
                "recon_losses": avg_recon,
                "kl_losses": avg_kl,
                "contrastive_losses": avg_contrastive
            }
            track_metrics(metrics, epoch_values)

        # Print single summary line on rank 0
        if rank == 0:
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f} | "
                f"Recon: {avg_recon:.2f} | "
                f"KL: {avg_kl:.2f} | "
                f"Contrastive: {avg_contrastive:.2f}"
            )

    # Save model and metrics (only rank 0)
    if rank == 0:
        os.makedirs("saved_models", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("saved_models", f"vae_{latent_dim}d_{num_epochs}epochs_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, "model.pth")
        save_model(vae, model_save_path)
        print(f"[Rank 0] Model saved to: {model_save_path}")

        if track_metrics_flag:
            plot_save_path = os.path.join(model_dir, "metrics.png")
            save_metrics_and_plots(plot_save_path, metrics, num_epochs)

        config_save_path = os.path.join(model_dir, "config.json")
        with open(config_save_path, "w") as f:
            json.dump({
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "beta": beta,
                "is_ddp": is_ddp,
                "data_dir": data_dir,
                "contrastive_margin": contrastive_margin,
                "contrastive_scale": contrastive_scale,
                "latent_dim": latent_dim
            }, f, indent=4)
        print(f"[Rank 0] Config saved to: {config_save_path}")

def main_worker(rank, world_size, args):
    """
    Main worker function for Distributed Data Parallel (DDP) training.
    """
    try:
        setup_ddp(rank, world_size)
        train(
            rank=rank,
            world_size=world_size,
            is_ddp=args.is_ddp,
            track_metrics_flag=args.track_metrics,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_dir=args.data_dir,
            image_size=tuple(args.image_size),
            beta=args.beta,
            contrastive_margin=args.contrastive_margin,
            contrastive_scale=args.contrastive_scale,
            latent_dim=args.latent_dim
        )
    except Exception as e:
        print(f"ERROR: Rank {rank} encountered an error: {e}")
    finally:
        # Print cleanup message only once, on rank 0
        if rank == 0:
            print("Cleaning up DDP")
        cleanup_ddp()

def get_args():
    """
    First parses for a configuration file and then sets defaults based on it.
    Command-line arguments override values in the config file.
    """
    config_parser = argparse.ArgumentParser(description="Train VAE Model", add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to JSON config file", default=None)
    args_config, remaining_args = config_parser.parse_known_args()

    config_defaults = {}
    if args_config.config:
        with open(args_config.config, "r") as f:
            config_defaults = json.load(f)

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--is_ddp", action="store_true", help="Use Distributed Data Parallel (DDP)")
    parser.add_argument("--track_metrics", action="store_true", help="Enable tracking of training metrics")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--image_size", type=int, nargs=2, help="Image size as two integers (width height)")
    parser.add_argument("--beta", type=float, help="Beta parameter for VAE loss")
    parser.add_argument("--contrastive_margin", type=float, help="Margin parameter for contrastive loss")
    parser.add_argument("--contrastive_scale", type=float, help="Scalar multiplier for contrastive loss")
    parser.add_argument("--latent_dim", type=int, help="Dimensionality of the latent space for the VAE")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_args)
    return args

if __name__ == "__main__":
    args = get_args()
    if args.is_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, 1, args.is_ddp, args.track_metrics, args.num_epochs, args.batch_size,
              args.learning_rate, args.data_dir, tuple(args.image_size),
              beta=args.beta, contrastive_margin=args.contrastive_margin,
              contrastive_scale=args.contrastive_scale, latent_dim=args.latent_dim)