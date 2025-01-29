import torch
import numpy as np
import matplotlib.pyplot as plt
from data import get_transforms, ImageDataset
from autoencoder.model import create_autoencoder
from autoencoder.utils import load_model
from torch.utils.data import DataLoader

def display_image(tensor, title="Image"):
    """
    Utility function to display an image tensor.
    """
    img = tensor.cpu().detach().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1] range
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

def evaluate_model(model_path, dataset_dir, batch_size=1, device="cpu"):
    """
    Loads a trained model, evaluates it on test data, and displays original and reconstructed images.

    :param model_path: Path to the saved model.
    :param dataset_dir: Directory containing the dataset.
    :param batch_size: Batch size for evaluation.
    :param device: Device to run the evaluation on ("cpu" or "cuda").
    """
    # Device setup
    device = torch.device(device)

    # Load test dataset
    test_transform = get_transforms(image_size=(2048, 2048), is_train=False)
    test_dataset = ImageDataset(data_dir=dataset_dir, transform=test_transform, split="test", test_size=0.2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    autoencoder = create_autoencoder()
    autoencoder = load_model(autoencoder, model_path, device)
    autoencoder.eval()

    # Get a sample image from the test set
    sample_img = next(iter(test_loader))
    sample_img = sample_img.to(device)

    # Reconstruct the image
    with torch.no_grad():
        reconstructed_img = autoencoder(sample_img)

    reconstructed_img = torch.clamp(reconstructed_img, 0, 1)

    # Display original and reconstructed images
    display_image(sample_img[0], title="Original Image")
    display_image(reconstructed_img[0], title="Reconstructed Image")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the evaluation ('cpu' or 'cuda')")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
