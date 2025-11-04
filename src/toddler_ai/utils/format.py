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
        return images


class MiniLMPreprocessor(ObservationPreprocessor):
    """MiniLM-based observation preprocessor.

    Uses pre-trained sentence transformers to encode text instructions
    into embeddings. The encoder can be fine-tuned during training.

    Supported models:
    - 'sentence-transformers/all-MiniLM-L6-v2' (default, 384-dim, 22.7M params)
    - 'sentence-transformers/paraphrase-MiniLM-L3-v2' (smaller, 384-dim, 17.4M params)
    - 'sentence-transformers/all-MiniLM-L12-v2' (larger, 384-dim, 33.4M params)
    """
    def __init__(self, model_name, obs_space=None,
                 model_name_minilm='sentence-transformers/all-MiniLM-L6-v2',
                 freeze_encoder=False):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv sync --extra language"
            )

        self.image_preproc = RawImagePreprocessor()
        self.freeze_encoder = freeze_encoder

        # Load MiniLM encoder
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f'Loading language model: {model_name_minilm}')
        self.minilm_encoder = SentenceTransformer(model_name_minilm)

        # Set trainability
        if freeze_encoder:
            logger.info('  Freezing encoder weights (projection only trainable)')
            for param in self.minilm_encoder.parameters():
                param.requires_grad = False
        else:
            logger.info('  Encoder weights TRAINABLE (full finetuning)')
            for param in self.minilm_encoder.parameters():
                param.requires_grad = True

        self.obs_space = {
            "image": 147,
            "minilm_emb": 384  # MiniLM output dimension (same for L3/L6/L12)
        }

    def __call__(self, obss, device=None):
        obs_ = DictList()

        # Process images
        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        # Process instructions with MiniLM
        if "minilm_emb" in self.obs_space.keys():
            # Extract mission texts
            missions = [obs["mission"] for obs in obss]

            # Compute MiniLM embeddings
            if self.freeze_encoder:
                # Frozen: compute outside grad context, then enable grad for projection
                with torch.no_grad():
                    embeddings = self.minilm_encoder.encode(
                        missions,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=device if device else 'cpu'
                    )
                # Clone and enable gradients for projection layer
                obs_.minilm_emb = embeddings.clone().detach().requires_grad_(True)
            else:
                # Trainable: compute WITH gradients - full finetuning!
                # NOTE: We can't use .encode() because it uses inference_mode internally.
                # Instead, we use the underlying model directly with tokenization.
                from transformers import AutoTokenizer

                # Get the tokenizer and model from SentenceTransformer
                tokenizer = self.minilm_encoder.tokenizer
                model = self.minilm_encoder[0].auto_model  # The transformer model

                # Tokenize
                encoded = tokenizer(
                    missions,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

                # Move to device
                if device:
                    encoded = {k: v.to(device) for k, v in encoded.items()}

                # Forward pass WITH gradients
                outputs = model(**encoded)

                # Mean pooling (same as SentenceTransformer)
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs[0]  # First element is last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

                # Normalize (same as SentenceTransformer)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # For RL: detach to avoid backward through graph multiple times (PPO epochs)
                # For IL: gradients flow through (single backward pass per batch)
                # We detach here, and projection layer will have gradients
                obs_.minilm_emb = embeddings.detach()

            if device and obs_.minilm_emb.device != device:
                obs_.minilm_emb = obs_.minilm_emb.to(device)

        return obs_


# Backward compatibility alias
MiniLMObssPreprocessor = MiniLMPreprocessor
