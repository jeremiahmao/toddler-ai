import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def load_model(model_name, raise_not_found=True, suffix=None):
    """Load model from disk.

    Args:
        model_name: Base name for the model directory
        raise_not_found: Whether to raise error if model not found
        suffix: Optional suffix for the filename (e.g., 'best' -> 'model_best.pt')
                If None, loads 'model.pt'
    """
    if suffix:
        # Load from same directory with different filename
        model_dir = get_model_dir(model_name)
        path = os.path.join(model_dir, f"model_{suffix}.pt")
    else:
        # Default: load model.pt
        path = get_model_path(model_name)

    try:
        device = utils.get_device()
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name, suffix=None):
    """Save model to disk.

    Args:
        model: The model to save
        model_name: Base name for the model directory
        suffix: Optional suffix for the filename (e.g., 'best' -> 'model_best.pt')
                If None, saves as 'model.pt'
    """
    if suffix:
        # Save to same directory with different filename
        model_dir = get_model_dir(model_name)
        path = os.path.join(model_dir, f"model_{suffix}.pt")
    else:
        # Default: save as model.pt
        path = get_model_path(model_name)

    utils.create_folders_if_necessary(path)
    torch.save(model, path)
