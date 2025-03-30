import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, distributed
import torch

#########################################
# Common Functions & Transforms
#########################################

def preprocess_image(image):
    """
    Normalize and convert an image to grayscale ('L') format.
    
    Args:
        image (PIL.Image.Image): Input PIL image.
    
    Returns:
        PIL.Image.Image: Preprocessed PIL image.
    """
    sample = np.array(image)
    sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    processed_image = Image.fromarray(sample).convert('L')
    return processed_image

def get_transforms(image_size=(224, 224), is_train=True):
    """
    Generate transformation pipeline for training or testing.
    
    Args:
        image_size (tuple[int, int]): Desired output image dimensions (height, width).
        is_train (bool): Whether to include data augmentation (True for training).
    
    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    if is_train:
        transform_list = [
            transforms.Lambda(preprocess_image),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]
    else:
        transform_list = [
            transforms.Lambda(preprocess_image),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]
    return transforms.Compose(transform_list)


#########################################
# Original Dataloader (TIFF files)
#########################################

class Data_Class(Dataset):
    """
    Custom Dataset for loading images with train-test splitting functionality.
    Reads image files (.TIF/.tif) from a directory.
    """
    def __init__(self, data_dir, transform=None, split="train", test_size=0.2, random_seed=42):
        self.data_dir = data_dir
        self.transform = transform

        # Get all image file paths
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.TIF', '.tif'))]

        # Split into train and test subsets
        train_files, test_files = train_test_split(
            self.image_files, test_size=test_size, random_state=random_seed
        )

        # Select the appropriate subset
        if split == "train":
            self.image_files = train_files
        elif split == "test":
            self.image_files = test_files
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

def create_dataloaders(data_dir, image_size=(224, 224), batch_size=8, test_size=0.2,
                       world_size=1, rank=0, is_ddp=False):
    """
    Create DataLoader objects for training and testing from a directory of image files.
    """
    image_size = tuple(image_size)
    
    train_dataset = Data_Class(
        data_dir=data_dir,
        transform=get_transforms(image_size=image_size, is_train=True),
        split="train",
        test_size=test_size
    )

    test_dataset = Data_Class(
        data_dir=data_dir,
        transform=get_transforms(image_size=image_size, is_train=False),
        split="test",
        test_size=test_size
    )

    # Distributed sampler for training dataset
    if is_ddp:
        train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True  # GPU efficient
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader


#########################################
# New NPZ Dataloader (Pre-extracted patches)
#########################################

class NPZ_Patch_Dataset(Dataset):
    """
    Custom Dataset for loading pre-extracted patches from a single NPZ file.
    The NPZ file must have a key 'patches' containing an array of shape (N, C, H, W).
    Train/test splitting is done on patch indices.
    """
    def __init__(self, npz_path, transform=None, split="train", test_size=0.2, random_seed=42):
        data = np.load(npz_path)
        self.all_patches = data["patches"]  # shape: (N, C, H, W)
        self.transform = transform

        indices = np.arange(self.all_patches.shape[0])
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_seed
        )
        if split == "train":
            self.indices = train_indices
        elif split == "test":
            self.indices = test_indices
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        patch = self.all_patches[real_idx]  # shape: (C, H, W)

        # Convert numpy patch to PIL image.
        if patch.shape[0] == 1:
            # Grayscale image: remove the channel dimension.
            patch_img = Image.fromarray(patch[0], mode="L")
        elif patch.shape[0] == 3:
            # Color image: transpose to (H, W, C)
            patch_img = Image.fromarray(np.transpose(patch, (1, 2, 0)), mode="RGB")
        else:
            raise ValueError(f"Unexpected number of channels: {patch.shape[0]}")

        if self.transform:
            patch_img = self.transform(patch_img)

        return patch_img

def create_npz_dataloaders(npz_path, image_size=(224, 224), batch_size=8, test_size=0.2,
                           world_size=1, rank=0, is_ddp=False):
    """
    Create DataLoader objects for training and testing from a single NPZ file containing patches.
    """
    train_dataset = NPZ_Patch_Dataset(
        npz_path=npz_path,
        transform=get_transforms(image_size=image_size, is_train=True),
        split="train",
        test_size=test_size
    )
    test_dataset = NPZ_Patch_Dataset(
        npz_path=npz_path,
        transform=get_transforms(image_size=image_size, is_train=False),
        split="test",
        test_size=test_size
    )
    
    if is_ddp:
        train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True  # for efficient GPU transfers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader