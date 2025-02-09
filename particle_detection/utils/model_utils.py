import os
import torch
import matplotlib.pyplot as plt

def save_model(model, path="autoencoder_model.pth"):
    """
    Saves the model's state dictionary to a specified file.

    Handles Distributed Data Parallel (DDP) models by accessing the underlying
    module's state dictionary if applicable.

    Args:
        model (torch.nn.Module): The PyTorch model to save (supports DDP and non-DDP models).
        path (str): Path to save the model file. Defaults to "autoencoder_model.pth".

    Returns:
        None
    """
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, path)
    print(f"Model saved to {path}")

def load_model(model, model_path, device):
    """
    Loads the model's state dictionary from a file, with support for DDP-trained models.

    Strips the "module." prefix from state dictionary keys if necessary to load
    DDP-trained models into a non-DDP model.

    Args:
        model (torch.nn.Module): The PyTorch model instance to load the weights into.
        model_path (str): Path to the saved model file.
        device (torch.device): The device to map the model (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model with the loaded state dictionary.
    """
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        # Handle DDP prefix
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {model_path}.")
    else:
        print(f"Model not found at {model_path}. Initializing a new model.")
    return model

def save_metrics_and_plots(filepath, losses, num_epochs, image_size):
    """
    Saves a plot of training loss metrics to a specified file.

    Creates the directory for the file if it does not exist.

    Args:
        filepath (str): Full path to save the plot (including the file name and extension).
        losses (list[float]): List of loss values for each epoch.
        num_epochs (int): Total number of training epochs.
        image_size (tuple[int, int]): Tuple representing the image size (height, width).

    Returns:
        None
    """
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"Loss for ae_{num_epochs}_{image_size[0]}x{image_size[1]}")
    plt.legend()
    plt.savefig(filepath)
    plt.close()
    print(f"Loss plot saved to {filepath}")