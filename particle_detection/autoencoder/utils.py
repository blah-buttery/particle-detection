import os
import torch

def save_model(model, path="autoencoder_model.pth"):
    """
    Save the model's state dictionary to a file, handling DDP models gracefully.
    
    :param model: The PyTorch model to save (supports DDP and non-DDP models).
    :param path: Path to save the model.
    """
    # Access the underlying model if wrapped with DDP
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, path)
    print(f"[DEBUG] Model saved to {path}")

def load_model(model, model_path, device):
    """
    Load the model's state dictionary from a file, with support for DDP-trained models.
    
    :param model: The PyTorch model instance to load the weights into.
    :param model_path: Path to the saved model file.
    :param device: The device to map the model (e.g., 'cpu' or 'cuda').
    """
    if os.path.exists(model_path):
        print(f"[DEBUG] Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle DDP prefix
        if any(key.startswith("module.") for key in state_dict.keys()):
            print("[DEBUG] Stripping 'module.' prefix from state_dict keys...")
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"[DEBUG] Model loaded successfully from {model_path}.")
    else:
        print(f"[DEBUG] Model not found at {model_path}. Initializing a new model.")
    return model

