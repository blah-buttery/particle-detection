def track_metrics(metrics, epoch_values, epoch=None):
    """
    Updates the metrics dictionary with the provided epoch values.

    Args:
        metrics (dict): A dictionary containing lists to track metrics 
                        (e.g., {"total_losses": [], "recon_losses": [], 
                                "kl_losses": [], "contrastive_losses": []}).
        epoch_values (dict): A dictionary containing the metric values for the current epoch.
                             For example: {"total_losses": 0.123, "recon_losses": 0.456, 
                                           "kl_losses": 0.078, "contrastive_losses": 0.012}.
        epoch (int, optional): The current epoch (for logging). Defaults to None.

    Raises:
        TypeError: If 'metrics' or 'epoch_values' is not a dictionary, or if any metric value is non-numeric.
        ValueError: If 'epoch' is provided and is not a non-negative integer.

    Example:
        metrics = {"total_losses": [], "recon_losses": [], "kl_losses": [], "contrastive_losses": []}
        epoch_values = {"total_losses": 0.123, "recon_losses": 0.456, "kl_losses": 0.078, "contrastive_losses": 0.012}
        track_metrics(metrics, epoch_values, epoch=1)
    """
    if not isinstance(metrics, dict):
        raise TypeError("Expected 'metrics' to be a dictionary.")
    if not isinstance(epoch_values, dict):
        raise TypeError("Expected 'epoch_values' to be a dictionary.")
    if epoch is not None and (not isinstance(epoch, int) or epoch < 0):
        raise ValueError("Epoch must be a non-negative integer.")

    for key, value in epoch_values.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected metric value for '{key}' to be a number.")
        metrics.setdefault(key, [])
        metrics[key].append(value)
    
    if epoch is not None:
        summary = " | ".join(f"{k.replace('_', ' ').capitalize()}: {v:.4f}" 
                             for k, v in epoch_values.items())
        print(f"Epoch {epoch}: {summary}")