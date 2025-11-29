import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def get_encoder_path(model_name, suffix=None):
    """Get path for bert-tiny encoder weights."""
    model_dir = get_model_dir(model_name)
    if suffix:
        return os.path.join(model_dir, f"encoder_{suffix}.pt")
    return os.path.join(model_dir, "encoder.pt")


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


def load_encoder(model_name, suffix=None):
    """Load bert-tiny encoder weights from disk.

    Returns the encoder state_dict if found, None otherwise.
    """
    encoder_path = get_encoder_path(model_name, suffix)
    try:
        device = utils.get_device()
        encoder_state = torch.load(encoder_path, map_location=device, weights_only=False)
        return encoder_state
    except FileNotFoundError:
        return None


def save_model(model, model_name, suffix=None, preprocessor=None):
    """Save model to disk.

    Args:
        model: The model to save
        model_name: Base name for the model directory
        suffix: Optional suffix for the filename (e.g., 'best' -> 'model_best.pt')
                If None, saves as 'model.pt'
        preprocessor: Optional preprocessor containing bert-tiny encoder to save
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

    # Also save bert-tiny encoder weights if preprocessor provided
    if preprocessor is not None and hasattr(preprocessor, 'minilm_encoder'):
        encoder_path = get_encoder_path(model_name, suffix)
        torch.save(preprocessor.minilm_encoder.state_dict(), encoder_path)
