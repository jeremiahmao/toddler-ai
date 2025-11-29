from abc import ABC, abstractmethod
import numpy
import torch
from toddler_ai.utils.dictlist import DictList

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def default_preprocess_obss(obss, device=None):
    """Default observation preprocessing function.

    Simply returns observations as-is without modification.
    Used as fallback when no custom preprocessor is provided.
    """
    return obss


class ObservationPreprocessor(ABC):
    """Base class for observation preprocessors.

    Preprocessors convert raw environment observations into tensors
    that neural networks can process.

    Subclasses must:
    - Implement __call__(obss, device=None)
    - Set self.obs_space in __init__
    """

    @abstractmethod
    def __call__(self, obss, device=None):
        """Process a batch of observations.

        Args:
            obss: List of observation dicts from the environment
            device: torch device to place tensors on

        Returns:
            DictList containing processed observation tensors
        """
        pass


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        # Normalize from 0-10 range to 0-1 (BabyAI encodes: object 0-10, color 0-5, state 0-2)
        images = images / 10.0
        return images


class MiniLMPreprocessor(ObservationPreprocessor):
    """Language model-based observation preprocessor.

    Uses bert-tiny (4.4M params, 128-dim) for efficient instruction encoding.
    Small enough to train with full gradients.
    """
    def __init__(self, model_name, obs_space=None,
                 model_name_minilm='prajjwal1/bert-tiny',
                 freeze_encoder=False,
                 encoder_from_model=None):
        self.image_preproc = RawImagePreprocessor()
        self.freeze_encoder = freeze_encoder

        # Load bert-tiny encoder
        import logging
        from transformers import AutoModel, AutoTokenizer
        from toddler_ai.utils.model import load_encoder
        logger = logging.getLogger(__name__)
        logger.info(f'Loading language model: {model_name_minilm}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_minilm)
        self.minilm_encoder = AutoModel.from_pretrained(model_name_minilm)

        encoder_params = sum(p.numel() for p in self.minilm_encoder.parameters())
        logger.info(f'  Encoder: {encoder_params/1e6:.1f}M params, 128-dim output')

        # Try to load trained encoder weights if they exist
        # Use encoder_from_model if specified (for loading from pretrained model),
        # otherwise use model_name (for resuming training)
        load_from = encoder_from_model if encoder_from_model is not None else model_name
        encoder_state = load_encoder(load_from)
        if encoder_state is not None:
            self.minilm_encoder.load_state_dict(encoder_state)
            if encoder_from_model:
                logger.info(f'  Loaded trained encoder weights from {encoder_from_model}')
            else:
                logger.info('  Loaded trained encoder weights from checkpoint')
        else:
            logger.info('  No trained encoder weights found, using pretrained bert-tiny')

        # Set trainability
        if freeze_encoder:
            logger.info('  Freezing encoder weights')
            for param in self.minilm_encoder.parameters():
                param.requires_grad = False
        else:
            logger.info('  Encoder weights TRAINABLE (full finetuning)')
            for param in self.minilm_encoder.parameters():
                param.requires_grad = True

        self.obs_space = {
            "image": 147,
            "minilm_emb": 128  # bert-tiny output dimension
        }

    def __call__(self, obss, device=None):
        obs_ = DictList()

        # Process images
        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        # Process instructions with bert-tiny
        if "minilm_emb" in self.obs_space.keys():
            # Extract mission texts
            missions = [obs["mission"] for obs in obss]

            # Tokenize
            encoded = self.tokenizer(
                missions,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            if device:
                encoded = {k: v.to(device) for k, v in encoded.items()}
                self.minilm_encoder = self.minilm_encoder.to(device)

            # Forward pass (with or without gradients based on freeze_encoder)
            if self.freeze_encoder:
                with torch.no_grad():
                    outputs = self.minilm_encoder(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].detach()  # CLS token
            else:
                outputs = self.minilm_encoder(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            obs_.minilm_emb = embeddings

        return obs_


# Backward compatibility alias
MiniLMObssPreprocessor = MiniLMPreprocessor
