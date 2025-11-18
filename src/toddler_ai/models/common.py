"""Common model components shared across architectures."""

import torch
import torch.nn as nn


def initialize_parameters(m):
    """Initialize network parameters.

    From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MiniLMProjection(nn.Module):
    """Projection layer for pre-computed MiniLM embeddings.

    Takes pre-computed MiniLM embeddings (384-dim) and projects them to instr_dim.
    This allows gradients to flow through the projection while keeping MiniLM frozen.

    The actual MiniLM encoding happens in the preprocessor to keep it outside
    the computation graph (frozen), but the projection is trainable.
    """
    def __init__(self, instr_dim=128, minilm_dim=384):
        super().__init__()
        self.minilm_dim = minilm_dim
        self.instr_dim = instr_dim

        # Trainable projection from MiniLM dims to instr_dim
        # This will receive gradients during backprop
        self.projection = nn.Linear(minilm_dim, instr_dim)
        self.projection.apply(initialize_parameters)

    def forward(self, minilm_embeddings):
        """Project MiniLM embeddings to instruction dimension.

        Args:
            minilm_embeddings: Pre-computed MiniLM embeddings (batch_size, 384)
                              These are frozen (no grad from MiniLM itself)
                              but treated as leaf tensors that can pass gradients
                              to the projection layer

        Returns:
            Tensor of shape (batch_size, instr_dim) with gradients enabled
        """
        # Project to instr_dim - this operation has gradients!
        return self.projection(minilm_embeddings)
