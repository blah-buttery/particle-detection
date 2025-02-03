import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim, nn, distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from particle_detection.data.dataset import ImageDataset, get_transforms
from particle_detection.autoencoder.model import create_autoencoder
from particle_detection.autoencoder.utils import save_model, load_model

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
    dataset_dir="images/normal",
    model_path=None
):
    if is_ddp:
        setup_ddp(rank, world_size)

    # Device configuration
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    # Load dataset
    print("[DEBUG] Loading dataset...")
    transform = get_transforms(image_size=(1024, 1024), is_train=True)
    dataset = ImageDataset(data_dir=dataset_dir, transform=transform, split="train", test_size=0.2)
    print(f"[DEBUG] Total dataset size: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"[DEBUG] Train size: {train_size}, Test size: {test_size}")

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("[DEBUG] Dataloaders prepared.")

    # Load model
    print("[DEBUG] Initializing model...")
    model = create_autoencoder().to(device)
    if model_path:
        print(f"[DEBUG] Loading model from {model_path}")
        model = load_model(model, model_path=model_path, device=device)

    if is_ddp:
        model = DDP(model, device_ids=[rank])
        print("[DEBUG] Model wrapped with DDP.")

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("[DEBUG] Loss function and optimizer initialized.")

    # Training loop
    print("[DEBUG] Starting training loop...")
    for epoch in range(num_epochs):
        print(f"[DEBUG] Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        if is_ddp:
            train_sampler.set_epoch(epoch)
        train_loss = 0.0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"[DEBUG] Rank {rank if is_ddp else 'Single'} - Epoch {epoch + 1} - Batch {batch_idx + 1}/{len(train_loader)}")

        train_loss /= len(train_loader)
        print(f"[DEBUG] Rank {rank if is_ddp else 'Single'}, Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

    # Save model
    if rank == 0 or not is_ddp:
        print(f"[DEBUG] Saving model to {model_path}")
        save_model(model, model_path)

    if is_ddp:
        cleanup_ddp()

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../saved_models/autoencoder.pth")
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("--dataset_dir", type=str, default="images/normal", help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to a pre-trained model for fine-tuning")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--is_ddp", action="store_true", help="Use Distributed Data Parallel (multi-GPU)")
    args = parser.parse_args()

    print(f"[DEBUG] Saving model to: {args.model_path}")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    if args.is_ddp and args.world_size > 1:
        print("[DEBUG] Launching DDP training...")
        torch.multiprocessing.spawn(
            train,
            args=(args.world_size, args.is_ddp, args.num_epochs, args.batch_size, args.learning_rate, args.dataset_dir, args.model_path),
            nprocs=args.world_size,
            join=True
        )
    else:
        print("[DEBUG] Launching single-device training...")
        train(
            rank=0,
            world_size=1,
            is_ddp=False,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset_dir=args.dataset_dir,
            model_path=args.model_path
        )
