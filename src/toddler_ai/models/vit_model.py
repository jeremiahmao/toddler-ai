"""
Vision Transformer (ViT) based Actor-Critic model for BabyAI.

Modern architecture using:
- ViT for vision encoding
- Self-attention for vision reasoning
- Cross-attention for vision-language grounding
- MiniLM for language understanding

Replaces the FiLM-based CNN architecture with pure attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def initialize_parameters(m):
    """Initialize network parameters."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class PatchEmbedding(nn.Module):
    """Convert image into sequence of patch embeddings.

    For BabyAI 7x7 images, we use patch_size=1 (each cell is a patch).
    """
    def __init__(self, image_size=7, patch_size=1, in_channels=3, embed_dim=128):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.apply(initialize_parameters)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        B, C, H, W = x.shape

        # Reshape to patches: [B, num_patches, patch_size*patch_size*C]
        # For patch_size=1: [B, H*W, C]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, self.num_patches, -1)  # [B, H*W, patch_size*patch_size*C]

        # Project to embedding dimension
        x = self.projection(x)  # [B, num_patches, embed_dim]

        # Add positional embeddings
        x = x + self.pos_embedding

        return x


class VisionSelfAttention(nn.Module):
    """Self-attention layer for vision patches to reason about spatial relationships.

    Allows patches to communicate with each other before language conditioning.
    """
    def __init__(self, dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Layer norm for stable training
        self.norm = nn.LayerNorm(dim)

        # MLP for additional processing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)

        self.apply(initialize_parameters)

    def forward(self, x):
        """
        Args:
            x: [B, N, dim] patch embeddings
        Returns:
            [B, N, dim] attended patch embeddings
        """
        B, N, C = x.shape

        # Residual connection + self-attention
        residual = x
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        # Residual
        x = x + residual

        # Residual + MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class CrossAttention(nn.Module):
    """Cross-attention between vision and language.

    Vision queries language to ground instructions in visual features.
    This is the key mechanism for vision-language grounding.
    """
    def __init__(self, dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Separate projections for query (vision) and key/value (language)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.apply(initialize_parameters)

    def forward(self, vision, language):
        """
        Args:
            vision: [B, N, dim] vision patch embeddings
            language: [B, dim] or [B, seq_len, dim] language embeddings
        Returns:
            [B, N, dim] language-grounded vision features
        """
        B, N, C = vision.shape

        # Handle both single-vector and sequence language embeddings
        if language.dim() == 2:
            language = language.unsqueeze(1)  # [B, 1, dim]

        residual = vision
        vision = self.norm(vision)

        # Compute Q (from vision), K, V (from language)
        q = self.q_proj(vision).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(language).reshape(B, -1, self.num_heads, self.head_dim)
        v = self.v_proj(language).reshape(B, -1, self.num_heads, self.head_dim)

        # Transpose for attention: [B, heads, N, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, seq_len]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection
        out = out + residual

        return out


class ViTACModel(nn.Module):
    """Vision Transformer based Actor-Critic model.

    Modern architecture:
    1. Image → ViT patch embeddings
    2. Vision self-attention (patches reason about each other)
    3. Cross-attention with language (vision queries language)
    4. Pool to single vector
    5. LSTM memory (optional)
    6. Actor/Critic heads
    """
    def __init__(
        self,
        obs_space,
        action_space,
        image_size=7,
        patch_size=1,
        embed_dim=128,
        memory_dim=128,
        use_memory=True,
        vit_depth=1,
        vit_heads=1,
        cross_attn_heads=1,
        dropout=0.1
    ):
        super().__init__()

        self.use_memory = use_memory
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim
        self.image_size = image_size

        # Stage 0: MiniLM projection (384-dim → embed_dim)
        # Only needed when using MiniLM language encoder
        if "minilm_emb" in obs_space:
            from toddler_ai.models.ac_model import MiniLMProjection
            self.minilm_projection = MiniLMProjection(instr_dim=embed_dim, minilm_dim=384)
        else:
            self.minilm_projection = None

        # Stage 1: Patch embedding (image → tokens)
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        # Stage 2: Vision self-attention (tokens reason about each other)
        self.vision_self_attn = nn.ModuleList([
            VisionSelfAttention(dim=embed_dim, num_heads=vit_heads, dropout=dropout)
            for _ in range(vit_depth)
        ])

        # Stage 3: Cross-attention (vision queries language)
        self.cross_attn = CrossAttention(
            dim=embed_dim,
            num_heads=cross_attn_heads,
            dropout=dropout
        )

        # Stage 4: Pool tokens to single vector
        self.pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

        # Stage 5: LSTM memory (optional)
        if use_memory:
            self.memory_rnn = nn.LSTMCell(embed_dim, memory_dim)
            self.embedding_size = memory_dim
        else:
            self.embedding_size = embed_dim

        # Stage 6: Actor and Critic heads
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize all parameters
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        if self.use_memory:
            return 2 * self.memory_dim
        return 0

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def _get_instr_embedding(self, instr=None, minilm_embeddings=None):
        """Get instruction embedding from pre-computed MiniLM embeddings.

        Args:
            instr: Not used (for compatibility with ACModel interface)
            minilm_embeddings: Pre-computed MiniLM embeddings tensor (batch_size, 384)

        Returns:
            Instruction embeddings (batch_size, embed_dim)
        """
        if minilm_embeddings is None:
            raise ValueError("ViT model requires pre-computed MiniLM embeddings")

        if self.minilm_projection is not None:
            return self.minilm_projection(minilm_embeddings)
        else:
            # Should not happen, but fallback for safety
            return minilm_embeddings

    def forward(self, obs, memory, instr_embedding=None):
        """
        Args:
            obs: Observation dict with:
                - image: [B, H, W, 3] image tensor
                - minilm_emb: [B, 384] language embedding from MiniLM
            memory: [B, memory_size] LSTM memory (if use_memory=True)
            instr_embedding: [B, embed_dim] optional pre-computed instruction embedding

        Returns:
            dict with:
                - dist: action distribution
                - value: state value
                - memory: updated memory
        """
        # Get instruction embedding (from MiniLM)
        if instr_embedding is None:
            # Check if obs has minilm_emb (handle DictList which can raise KeyError on hasattr)
            try:
                minilm_emb = obs.minilm_emb
            except (AttributeError, KeyError):
                minilm_emb = None

            if minilm_emb is not None:
                # Project MiniLM embedding (384-dim) to embed_dim (128-dim)
                if self.minilm_projection is not None:
                    instr_embedding = self.minilm_projection(minilm_emb)
                else:
                    # Fallback: use MiniLM embedding directly (requires embed_dim=384)
                    instr_embedding = minilm_emb
            else:
                raise ValueError("No instruction embedding provided. ViT model requires MiniLM embeddings.")

        # Stage 1: Convert image to patch embeddings
        # Input: [B, H, W, C] → Transpose to [B, C, H, W]
        x = obs.image.permute(0, 3, 1, 2).float()  # [B, 3, H, W]

        # Normalize to [0, 1]
        x = x / 255.0

        # Get patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Stage 2: Vision self-attention (patches reason spatially)
        for layer in self.vision_self_attn:
            x = layer(x)  # [B, num_patches, embed_dim]

        # Stage 3: Cross-attention with language
        x = self.cross_attn(x, instr_embedding)  # [B, num_patches, embed_dim]

        # Stage 4: Pool to single vector (mean pooling)
        x = x.mean(dim=1)  # [B, embed_dim]
        x = self.pool(x)  # [B, embed_dim]

        # Stage 5: LSTM memory (optional)
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x
            memory = torch.zeros(x.size(0), 0).to(x.device)

        # Stage 6: Actor and Critic
        action_logits = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(action_logits, dim=1))

        value = self.critic(embedding)
        value = value.squeeze(1)

        return {
            'dist': dist,
            'value': value,
            'memory': memory,
            'extra_predictions': {}
        }
