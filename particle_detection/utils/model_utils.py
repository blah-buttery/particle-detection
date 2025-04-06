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

def save_metrics_and_plots(filepath, metrics, num_epochs):
    """
    Save a single figure with separate subplots for each metric.
    
    Each metric in the dictionary is plotted in its own subplot so that differences 
    in magnitude are easier to visualize.
    
    Args:
        filepath (str): Full path (including filename and extension) to save the plot.
        metrics (dict): Dictionary containing lists of loss values for each epoch.
                        Expected keys: 'total_losses', 'recon_losses', 'kl_losses', 'contrastive_losses'.
        num_epochs (int): Total number of training epochs.
    
    Returns:
        None
    """
    import os
    import matplotlib.pyplot as plt

    # Ensure the directory exists.
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)

    epochs = range(1, num_epochs + 1)

    # Create a 2x2 grid for the subplots.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    # Plot each metric in its own subplot.
    for ax, (key, values) in zip(axs, metrics.items()):
        ax.plot(epochs, values, marker='o', linestyle='-')
        ax.set_title(key.replace("_", " ").capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
    
    fig.suptitle(f"Training Metrics over {num_epochs} Epochs")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filepath)
    plt.close()
    #print(f"Subplots saved to {filepath}")