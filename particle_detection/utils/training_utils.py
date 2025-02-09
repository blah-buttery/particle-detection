def track_metrics(metrics, loss, epoch=None):
    """
    Updates the metrics dictionary with the provided loss.

    Args:
        metrics (dict): A dictionary containing lists to track metrics (e.g., {"loss": []}).
        loss (float): The loss value to append to the metrics.
        epoch (int, optional): The current epoch, used for logging or debugging. Defaults to None.

    Raises:
        TypeError: If 'metrics' is not a dictionary or 'loss' is not numeric.
        ValueError: If 'epoch' is provided and is not a non-negative integer.

    Example:
        metrics = {"loss": []}
        track_metrics(metrics, loss=0.123, epoch=1)
    """
    # Validate inputs
    if not isinstance(metrics, dict):
        raise TypeError("Expected 'metrics' to be a dictionary.")
    if not isinstance(loss, (float, int)):
        raise TypeError("Expected 'loss' to be a float or int.")
    if epoch is not None and (not isinstance(epoch, int) or epoch < 0):
        raise ValueError("Epoch must be a non-negative integer.")

    # Ensure the "loss" key exists in the metrics dictionary
    metrics.setdefault("loss", [])
    metrics["loss"].append(loss)

    # Log progress
    if epoch is not None:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")