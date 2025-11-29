"""
Unified Concept Space Vision Transformer (ViT) with Predictive Processing.

Cognitive architecture inspired by human toddler learning:
- All modalities (vision, language, episodic memory) in unified 256-dim concept space
- Episodic memory: stores full 256-dim pooled representations from past timesteps
- Each memory entry captures consolidated experience (goal + vision + context)
- Self-attention learns to retrieve relevant past experiences
- Predictive processing: model predicts next observations conditioned on action (supplemental)
- Specialized MLP output heads for action selection and value estimation

Mental model: Inputs (multimodal) → Unified Concept Space (understanding) → Outputs (decisions).
The unified space is for perception and understanding; actions are task-specific outputs.
Memory stores consolidated experiences as full representations, not just action indices.
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
    Projects to unified concept space (256-dim).
    """
    def __init__(self, image_size=7, patch_size=1, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

        # Learnable spatial position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.apply(initialize_parameters)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] image tensor
        Returns:
            [B, num_patches, embed_dim] patch embeddings with spatial positions
        """
        B, C, H, W = x.shape

        # Reshape to patches: [B, num_patches, patch_size*patch_size*C]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, self.num_patches, -1)  # [B, H*W, patch_size*patch_size*C]

        # Project to embedding dimension
        x = self.projection(x)  # [B, num_patches, embed_dim]

        # Add spatial positional embeddings
        x = x + self.pos_embedding

        return x


class UnifiedSelfAttention(nn.Module):
    """Self-attention over unified context buffer.

    All tokens (goal, action history, vision patches) attend to each other.
    Temporal and spatial position encodings guide attention patterns.
    """
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
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
            x: [B, N, dim] unified context buffer
        Returns:
            [B, N, dim] attended context
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


class UnifiedViTACModel(nn.Module):
    """Vision Transformer with Unified Concept Space and Predictive Processing.

    Architecture:
    1. Image → patch embeddings [49, 256]
    2. MiniLM goal → projected to [256]
    3. Action history → embedded via action_embedding [history_len, 256]
    4. Concatenate [goal, action_history, vision_patches] → buffer [~60, 256]
    5. Self-attention over entire buffer (unified concept space for understanding)
    6. Pool buffer → state_concept [256]
    7. Outputs (specialized heads for task-specific decisions):
       - Action: MLP actor head (256 → 128 → 7) → action logits
       - Value: MLP critic head (256 → 128 → 1) → value
       - Vision prediction (supplemental, 0.01 coef): [49, 256] - predicts next observation
    """
    def __init__(
        self,
        obs_space,
        action_space,
        image_size=7,
        patch_size=1,
        embed_dim=256,  # Unified concept space dimension
        use_memory=True,  # Use episodic memory
        attn_depth=2,  # Number of attention layers
        attn_heads=4,  # Number of attention heads
        dropout=0.1,
        history_length=64,  # Number of past experiences to remember (episodic memory slots)
        vision_pred_coef=0.01  # Coefficient for vision prediction loss (supplemental, weak)
    ):
        super().__init__()

        self.use_memory = use_memory
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.num_actions = action_space.n
        self.history_length = history_length if use_memory else 0
        self.vision_pred_coef = vision_pred_coef

        # Language model projection (128-dim bert-tiny → embed_dim)
        if "minilm_emb" in obs_space:
            # Get embedding dimension from obs_space
            instr_dim = obs_space.get("minilm_emb", 128)
            self.minilm_projection = nn.Linear(instr_dim, embed_dim)
        else:
            self.minilm_projection = None

        # Patch embedding (image → tokens in concept space)
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # Action embeddings (for vision prediction conditioning)
        self.action_embeddings = nn.Embedding(self.num_actions, embed_dim)

        # Temporal position embeddings for episodic memory
        if use_memory:
            self.temporal_pos_embeddings = nn.Parameter(
                torch.randn(1, history_length, embed_dim)
            )

        # Unified self-attention over context buffer
        self.context_attention = nn.ModuleList([
            UnifiedSelfAttention(dim=embed_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(attn_depth)
        ])

        # Pool context to single state vector
        self.pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        # Standard MLP for stable action selection with proper gradient flow
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, self.num_actions)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Predictive heads
        # Vision predictor: predict next patch embeddings
        # This is supplemental - helps learn better representations by predicting state transitions
        # Should use weak coefficient (0.01-0.05) to not interfere with RL objective
        self.vision_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, self.num_patches * embed_dim)
        )

        # Initialize all parameters
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        """Memory size for compatibility with existing training code.

        Memory stores full 256-dim representations for each timestep.
        Shape: [B, history_length * embed_dim] flattened for storage.
        """
        if self.use_memory:
            return self.history_length * self.embed_dim
        return 0

    @property
    def semi_memory_size(self):
        """For compatibility - returns flattened memory size."""
        return self.history_length * self.embed_dim if self.use_memory else 0

    def forward(self, obs, memory, instr_embedding=None, action=None):
        """
        Args:
            obs: Observation dict with:
                - image: [B, H, W, 3] image tensor
                - minilm_emb: [B, 128] language embedding from bert-tiny
            memory: [B, history_length * embed_dim] episodic memory (flattened representations)
            instr_embedding: Optional pre-computed instruction embedding
            action: Optional [B] action indices for vision prediction conditioning

        Returns:
            dict with:
                - dist: action distribution
                - value: state value
                - memory: same memory (updated externally with new_memory)
                - new_memory: [B, embed_dim] the new memory entry to store
                - extra_predictions: dict with vision_pred (conditioned on action if provided)
        """
        batch_size = obs.image.size(0)

        # 1. Get goal embedding from MiniLM
        if instr_embedding is None:
            try:
                minilm_emb = obs.minilm_emb
            except (AttributeError, KeyError):
                minilm_emb = None

            if minilm_emb is not None:
                if self.minilm_projection is not None:
                    goal_embedding = self.minilm_projection(minilm_emb)  # [B, embed_dim]
                else:
                    goal_embedding = minilm_emb
            else:
                raise ValueError("No instruction embedding provided")
        else:
            # instr_embedding was provided - check if it needs projection
            # If it's not already embed_dim, project it
            if instr_embedding.size(-1) != self.embed_dim and self.minilm_projection is not None:
                goal_embedding = self.minilm_projection(instr_embedding)  # [B, embed_dim]
            else:
                goal_embedding = instr_embedding

        goal_token = goal_embedding.unsqueeze(1)  # [B, 1, embed_dim]

        # 2. Get vision patch embeddings
        # Input: [B, H, W, C] → Transpose to [B, C, H, W]
        # Note: images are pre-normalized in format.py (values 0-10 → 0-1)
        image = obs.image.permute(0, 3, 1, 2).float()
        vision_patches = self.patch_embed(image)  # [B, num_patches, embed_dim]

        # Store current vision for prediction target
        current_vision_patches = vision_patches.detach()

        # 3. Get episodic memory representations
        if self.use_memory and memory.size(1) > 0:
            # Memory is flattened: [B, history_length * embed_dim]
            # Reshape to [B, history_length, embed_dim]
            episodic_memory = memory.view(batch_size, self.history_length, self.embed_dim)
            # Add temporal position encodings
            episodic_memory = episodic_memory + self.temporal_pos_embeddings
        else:
            # No memory: empty sequence
            episodic_memory = torch.zeros(batch_size, 0, self.embed_dim).to(goal_token.device)

        # 4. Build unified context buffer
        # Concatenate: [goal, episodic_memory, vision_patches]
        context = torch.cat([goal_token, episodic_memory, vision_patches], dim=1)
        # Shape: [B, 1 + history_length + num_patches, embed_dim]

        # 5. Self-attention over context
        for attn_layer in self.context_attention:
            context = attn_layer(context)

        # 6. Pool to state concept
        # Mean pooling over all tokens
        state_concept = context.mean(dim=1)  # [B, embed_dim]
        state_concept = self.pool(state_concept)  # [B, embed_dim]

        # 7. Generate outputs

        # Action selection: standard MLP actor head
        # This provides stable gradients and proper entropy
        action_logits = self.actor(state_concept)  # [B, num_actions]
        dist = Categorical(logits=action_logits)

        # Value function
        value = self.critic(state_concept).squeeze(1)  # [B]

        # Predictive outputs
        # Vision prediction: what will I see next, conditioned on action?
        # This is supplemental - helps learn better state representations
        if action is not None:
            # Condition prediction on action taken
            action_emb = self.action_embeddings(action.long())  # [B, embed_dim]
            pred_input = state_concept + action_emb  # Add action context
        else:
            pred_input = state_concept

        vision_pred_flat = self.vision_predictor(pred_input)  # [B, num_patches * embed_dim]
        vision_pred = vision_pred_flat.view(batch_size, self.num_patches, self.embed_dim)  # [B, num_patches, embed_dim]

        # Package predictions for loss computation
        extra_predictions = {
            'vision_pred': vision_pred,
            'current_vision': current_vision_patches
        }

        return {
            'dist': dist,
            'value': value,
            'memory': memory,  # Memory stays same during forward (updated externally)
            'new_memory': state_concept.detach(),  # New memory entry to store [B, embed_dim]
            'extra_predictions': extra_predictions
        }
