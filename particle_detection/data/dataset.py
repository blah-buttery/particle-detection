import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, split="train", test_size=0.2, random_seed=42):
        """
        Dataset loader with train-test split.

        :param data_dir: Path to the directory containing images.
        :param transform: Transform to apply to the images.
        :param split: 'train' or 'test' to indicate the subset.
        :param test_size: Proportion of the dataset to use for testing.
        :param random_seed: Random seed for reproducibility.
        """
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

def get_transforms(image_size=(2048, 2048), is_train=True):
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
