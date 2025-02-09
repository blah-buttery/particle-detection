import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, distributed

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images with train-test splitting functionality.

    Args:
        data_dir (str): Path to the directory containing images.
        transform (callable, optional): Transformations to apply to the images. Defaults to None.
        split (str): Subset to load, either 'train' or 'test'. Defaults to 'train'.
        test_size (float): Proportion of the dataset to use for testing. Defaults to 0.2.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
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
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieve an image by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            PIL.Image.Image: Transformed image.
        """
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image


def preprocess_image(image):
    """
    Normalize and convert an image to RGB format.

    Args:
        image (PIL.Image.Image): Input PIL image.

    Returns:
        PIL.Image.Image: Preprocessed PIL image.
    """
    sample = np.array(image)
    sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    processed_image = Image.fromarray(sample).convert('RGB')
    return processed_image


def get_transforms(image_size=(224, 224), is_train=True):
    """
    Generate transformations for training or testing.

    Args:
        image_size (tuple[int, int]): Desired image dimensions (height, width). Defaults to (2048, 2048).
        is_train (bool): Whether to include data augmentation (True for training). Defaults to True.

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


def create_dataloaders(data_dir, image_size=(224, 224), batch_size=8, test_size=0.2, world_size=1, rank=0, is_ddp=False):
    """
    Create PyTorch DataLoader objects for training and testing, with optional DDP support.

    Args:
        data_dir (str): Path to the directory containing images.
        image_size (tuple[int, int]): Desired image dimensions (width, height). Defaults to (224, 224).
        batch_size (int): Batch size for the DataLoader. Defaults to 8.
        test_size (float): Proportion of the dataset to use for testing. Defaults to 0.2.
        world_size (int): Number of processes in DDP. Defaults to 1.
        rank (int): Rank of the current process in DDP. Defaults to 0.
        is_ddp (bool): Whether to use Distributed Data Parallel (DDP). Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader]: Tuple containing the train and test DataLoaders.
    """
    image_size = tuple(image_size)
    
    train_dataset = ImageDataset(
        data_dir=data_dir,
        transform=get_transforms(image_size=image_size, is_train=True),
        split="train",
        test_size=test_size
    )

    test_dataset = ImageDataset(
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
        sampler=train_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader