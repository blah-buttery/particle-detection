import os
import torch

def save_model(model, path="autoencoder_model.pth"):
    """
    Save the model's state dictionary to a file.

    :param model: The PyTorch model to save.
    :param path: Path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, model_path, device):
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found at {model_path}. Initializing a new model.")
    return model
