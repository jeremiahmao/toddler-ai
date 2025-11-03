import os
import json
import numpy
import re
import torch
from toddler_ai.utils.dictlist import DictList

from .. import utils

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


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None, load_vocab_from=None):
        self.image_preproc = RawImagePreprocessor()
        self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class MiniLMObssPreprocessor(object):
    """Observation preprocessor for MiniLM-based models.

    Computes MiniLM embeddings for instructions with full gradient support.
    The encoder is now TRAINABLE - gradients flow all the way through.

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
