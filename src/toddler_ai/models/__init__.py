"""Neural network models for Toddler AI.

Available architectures:
- ViTACModel: Vision Transformer with cross-attention (baseline, 486K params)
- UnifiedViTACModel: Unified concept space ViT with predictive processing (RECOMMENDED, 8.4M params)

Why both exist:
- vit: Smaller, faster baseline for quick experiments and resource-constrained environments
- unified_vit: State-of-the-art architecture with better performance but more compute
"""

from __future__ import annotations

from toddler_ai.models.vit_model import ViTACModel
from toddler_ai.models.unified_vit_model import UnifiedViTACModel

__all__ = ['ViTACModel', 'UnifiedViTACModel']
