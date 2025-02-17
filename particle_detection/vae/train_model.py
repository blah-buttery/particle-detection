import os
import json
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from particle_detection.data.data_pipeline import create_dataloaders
from particle_detection.vae.model import create_vae
from particle_detection.utils.training_utils import track_metrics
from particle_detection.utils.ddp_utils import setup_ddp, cleanup_ddp
from particle_detection.utils.model_utils import save_model, save_metrics_and_plots

def train(rank, world_size, is_ddp, track_metrics_flag, num_epochs, batch_size, learning_rate, data_dir, image_size=(1024, 1024)):
    """
    Train the Variational Autoencoder (VAE) model.

    Args:
        rank (int): Rank of the current process in DDP.
        world_size (int): Total number of processes participating in DDP.
        is_ddp (bool): Whether Distributed Data Parallel (DDP) is enabled.
        track_metrics_flag (bool): Whether to track and save training metrics.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        data_dir (str): Path to the dataset directory.
        image_size (tuple): Tuple specifying the image size (height, width).
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        is_ddp=is_ddp
    )

    # Load VAE model with transferred AE weights
    vae = create_vae("./saved_models/ae_300_1024x1024/model.pth")
    vae.to(device)
    
    beta=1.5

    if is_ddp:
        vae = DDP(vae, device_ids=[rank])

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    def vae_loss(recon_x, x, mu, logvar, beta=1.0):
        """VAE loss function: Reconstruction loss + KL Divergence"""
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + (beta * kl_div)

    metrics = {"loss": []}

    for epoch in range(num_epochs):
        if is_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        vae.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            recon_x, mu, logvar = vae(batch)
            loss = vae_loss(recon_x, batch, mu, logvar, beta=beta)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if track_metrics_flag:
            track_metrics(metrics, avg_loss, epoch + 1)

    if rank == 0:
        # Save model
        os.makedirs("saved_models", exist_ok=True)
        model_dir = os.path.join("saved_models", f"vae_{num_epochs}_{image_size[0]}x{image_size[1]}")
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, "model.pth")
        save_model(vae, model_save_path)

        # Save metrics
        if track_metrics_flag:
            plot_save_path = os.path.join(model_dir, "metrics.png")
            save_metrics_and_plots(
                plot_save_path, 
                metrics["loss"], 
                num_epochs, 
                image_size
            )

        for img in train_loader:
          color_mode = "grayscale" if img.shape[1] == 1 else "rgb"
          break

        # Save training config
        config_save_path = os.path.join(model_dir, "config.json")
        with open(config_save_path, "w") as f:
            json.dump({
                "image_size": image_size,
                "batch_size": batch_size,
                "image_type": color_mode,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "beta": beta,
                "ddp": is_ddp,
                "data_dir": data_dir
            }, f, indent=4)

def main_worker(rank, world_size, args):
    """
    Main worker function for Distributed Data Parallel (DDP) training.

    Args:
        rank (int): Rank of the current process in DDP.
        world_size (int): Total number of processes participating in DDP.
        args (argparse.Namespace): Parsed command-line arguments.
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
        )
    except Exception as e:
        print(f"ERROR: Rank {rank} encountered an error: {e}")
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    """
    Entry point for training the Variational Autoencoder (VAE).

    Command-line Arguments:
        --is_ddp: Use Distributed Data Parallel (DDP) training.
        --track_metrics: Enable tracking of training metrics.
        --num_epochs: Number of training epochs.
        --batch_size: Batch size for training.
        --learning_rate: Learning rate for the optimizer.
        --data_dir: Path to the dataset directory.
        --image_size: Image size as two integers (height, width).
    """
    parser = argparse.ArgumentParser(description="Train VAE Model")
    parser.add_argument("--is_ddp", action="store_true", help="Use Distributed Data Parallel (DDP)")
    parser.add_argument("--track_metrics", action="store_true", help="Enable tracking of training metrics")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="Image size as two integers (width height)")
    args = parser.parse_args()

    if args.is_ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, 1, args.is_ddp, args.track_metrics, args.num_epochs, args.batch_size, args.learning_rate, args.data_dir, tuple(args.image_size))

