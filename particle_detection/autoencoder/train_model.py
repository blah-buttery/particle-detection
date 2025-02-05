import os
import torch
from torch.utils.data import DataLoader, random_split, distributed
from torch import optim, nn, distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from particle_detection.data.dataset import ImageDataset, get_transforms, create_dataloaders
from particle_detection.autoencoder.model import create_autoencoder, create_vae
from particle_detection.autoencoder.utils import save_model, load_model
import argparse

def setup_ddp(rank, world_size):
    """
    Sets up DDP environment.
    """
    print(f"[DEBUG] Setting up DDP for rank {rank} with world size {world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    """
    Cleans up DDP environment.
    """
    print("[DEBUG] Cleaning up DDP")
    dist.destroy_process_group()

def train(
    rank,
    world_size,
    is_ddp,
    num_epochs=10,
    batch_size=16,
    learning_rate=0.0001,
    data_dir="/path/to/dataset",
    model_save_path="/path/to/save/model.pt",
    model_fn=None,
    criterion_fn=None
):
    """
    Train the model.

    :param rank: Rank of the current process (for DDP).
    :param world_size: Total number of processes (for DDP).
    :param is_ddp: Whether to use DDP.
    :param num_epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param data_dir: Path to the dataset directory.
    :param model_save_path: Path to save the trained model.
    :param model_fn: Function to create the model.
    :param criterion_fn: Function to create the loss criterion.
    """
    # Setup device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Get transforms
    transform = get_transforms(image_size=(1024, 1024), is_train=True)

    # Create dataset
    dataset = ImageDataset(data_dir, transform=transform)
    
    # Apply DistributedSampler if using DDP
    if is_ddp:
        train_sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False  # Shuffling is handled by the sampler
    else:
        train_sampler = None
        shuffle = True
    
    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler)
    
    # Validate batch size
    if batch_size > len(train_loader.dataset):
        print(f"[WARNING] Batch size ({batch_size}) exceeds dataset size ({len(train_loader.dataset)}). Adjusting batch size.")
        batch_size = len(train_loader.dataset)

    # Initialize model
    if model_fn is None:
        raise ValueError("A model creation function (model_fn) must be provided. Available types: autoencoder, vae, cnn.")
    model = model_fn()
    model.to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    if criterion_fn is None:
        raise ValueError("A loss criterion function (criterion_fn) must be provided.")
    if not callable(criterion_fn):
        raise TypeError("Provided criterion_fn must be callable. Ensure you are passing a valid loss function.")
    criterion = criterion_fn()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        if is_ddp:
            # Set epoch to ensure proper shuffling across all processes in DDP
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"[INFO] Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

    # Save the model
    if rank == 0 or not is_ddp:
        save_model(model, model_save_path)
        print(f"[INFO] Model saved to {model_save_path}")

    # Cleanup
    if is_ddp:
        cleanup_ddp()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Total number of processes (for DDP)")
    parser.add_argument("--is_ddp", action="store_true", help="Use Distributed Data Parallel (DDP)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--model_type", type=str, required=True, help="Type of model to train (autoencoder, vae, cnn)")
    args = parser.parse_args()

    model_map = {
        "autoencoder": create_autoencoder,
        "vae": create_vae,
        "cnn": create_cnn
    }
    
    criterion_map = {
        "autoencoder": nn.MSELoss,
        "vae": nn.BCELoss,
        "cnn": nn.CrossEntropyLoss
    }
    
    if args.model_type not in model_map:
        raise ValueError(f"Unsupported model type: {args.model_type}. Available types: autoencoder, vae, cnn.")

    model_fn = model_map[args.model_type]
    criterion_fn = criterion_map[args.model_type]

    train(0, args.world_size, args.is_ddp, args.num_epochs, args.batch_size, args.learning_rate, args.data_dir, args.model_save_path, model_fn, criterion_fn)