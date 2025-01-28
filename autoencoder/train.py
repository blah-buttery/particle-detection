import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from autoencoder.model import build_autoencoder
from data.dataset import Load_data, Preprocess

def setup(rank, world_size):
    """Setup for distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup for distributed training."""
    dist.destroy_process_group()

def train_autoencoder(rank, world_size, epochs=50, batch_size=4, learning_rate=0.0001):
    """Train the autoencoder on multiple devices."""
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset and Dataloader
    transform = transforms.Compose([
        Preprocess(),
        transforms.Resize((2048, 2048)),
        transforms.ToTensor(),
    ])
    dataset = Load_data(root_dir="images", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build and wrap model
    autoencoder = build_autoencoder(pretrained=True, device=device)
    model = DDP(autoencoder, device_ids=[rank]) if world_size > 1 else autoencoder

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

    # Save model on rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), "saved_models/autoencoder_model.pth")
        print("Model saved successfully!")

    cleanup()

def run_training():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        mp.spawn(train_autoencoder, args=(world_size,), nprocs=world_size, join=True)
    else:
        train_autoencoder(rank=0, world_size=1)
