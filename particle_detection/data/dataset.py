import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_files, data_dir, transform=None):
        """
        Simple Dataset class for loading images.

        :param image_files: List of image file names.
        :param data_dir: Path to the directory containing images.
        :param transform: Transform to apply to the images.
        """
        self.image_files = image_files
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def preprocess_image(image):
    """
    Normalize and convert the image to RGB format.

    :param image: Input PIL image.
    :return: Preprocessed PIL image.
    """
    sample = np.array(image)
    sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    processed_image = Image.fromarray(sample).convert('RGB')
    return processed_image


def get_transforms(image_size=(224, 224), is_train=True):
    """
    Simple function to get transformations for training or testing.

    :param image_size: Tuple of desired image dimensions (height, width).
    :param is_train: Whether to include augmentation (True for training, False for testing).
    :return: A torchvision.transforms.Compose object.
    """
    transform_list = [
        transforms.Lambda(preprocess_image),  # Integrate preprocessing here
        transforms.Resize(image_size),
        transforms.ToTensor()
    ]

    if is_train:
        # Add augmentations only for training
        transform_list.extend([
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    return transforms.Compose(transform_list)


def load_image_file_paths(data_dir, extensions=(".tif", ".TIF")):
    """
    List all image file paths in the directory with the given extensions.

    :param data_dir: Path to the directory containing images.
    :param extensions: Tuple of allowed image file extensions.
    :return: List of image file names.
    """
    return [f for f in os.listdir(data_dir) if f.endswith(extensions)]


def split_dataset(image_files, test_size=0.2, random_seed=42):
    """
    Split image file names into training and testing sets.

    :param image_files: List of image file names.
    :param test_size: Proportion of the dataset to use for testing.
    :param random_seed: Random seed for reproducibility.
    :return: Tuple of (train_files, test_files).
    """
    return train_test_split(image_files, test_size=test_size, random_state=random_seed)


def create_dataloaders(data_dir, transform, batch_size, test_size=0.2):
    """
    Create PyTorch DataLoader objects for training and testing.

    :param data_dir: Path to the directory containing images.
    :param transform: Transform to apply to the images.
    :param batch_size: Batch size for DataLoaders.
    :param test_size: Proportion of the dataset to use for testing.
    :return: Tuple of (train_loader, test_loader).
    """
    # Step 1: Load all image file paths
    image_files = load_image_file_paths(data_dir)

    # Step 2: Split into train and test sets
    train_files, test_files = split_dataset(image_files, test_size=test_size)

    # Step 3: Create Dataset objects
    train_dataset = ImageDataset(train_files, data_dir, transform=transform)
    test_dataset = ImageDataset(test_files, data_dir, transform=transform)

    # Step 4: Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
