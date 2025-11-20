# Toddler AI

A research platform for grounded language learning with unified concept space architectures, built on BabyAI/Minigrid environments.

## Overview

Toddler AI is a cognitive architecture research platform featuring:

- **Unified Concept Space ViT** - Novel architecture where vision, language, and action history share a 256-dim embedding space
- **BabyAI Environments** from [Minigrid](https://github.com/Farama-Foundation/Minigrid) - Grid-world tasks with natural language instructions
- **bert-tiny Language Encoder** - Lightweight 4.4M param encoder with full gradient flow
- **Modern Tooling** - Python 3.10+, PyTorch 2.0+, Gymnasium, uv package manager

## Architecture

### Unified Concept Space ViT (12.7M params)

```
Goal (bert-tiny) → [256] ─────────────────┐
Episodic Memory → [64, 256] ──────────────┼─→ Self-Attention [~114, 256] → Pool → [256]
Vision (patches) → [49, 256] ─────────────┘                                        ↓
                                                                          Actor + Critic MLPs
```

**Parameter Breakdown:**
- bert-tiny encoder: 4.4M params (128-dim output)
- Projection layer: 33K params (128 → 256)
- Patch embedding: 768 params (7×7 → 49 patches × 256-dim)
- Temporal position embeddings: 16K params (64 memory slots × 256-dim)
- Action embeddings: 1.8K params (7 actions × 256-dim, for vision prediction)
- Self-attention: 2.6M params (2 layers, 4 heads)
- Pool layer: 65K params
- Actor head: 33K params (256 → 128 → 7)
- Critic head: 33K params (256 → 128 → 1)
- Vision predictor: 3.2M params (supplemental, predicts next observation)

**Key Design:**
- All modalities encoded in shared 256-dim concept space
- Episodic memory: 64 full 256-dim representations of past experiences
- Each memory entry captures consolidated experience (goal + vision + context)
- Self-attention learns to retrieve relevant past experiences
- Memory updated externally with shift-left-and-add pattern
- Action-conditioned vision prediction (predicts next observation given action)
- Specialized MLP heads for action selection (not attention-based)

### Training

**Differential Learning Rates:**
- bert-tiny: 5e-6 (moderate inertia, 4.4M params)
- Model components: 1.5e-5 (medium inertia, 8.2M params)
- Task-specific heads: 5e-5 (trains freely, 133K params)

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/jeremiahmao/toddler-ai.git
cd toddler-ai
uv sync
```

## Quick Start

### 1. Generate Demonstrations

```bash
uv run python scripts/make_demos.py \
    --env BabyAI-GoToRedBallGrey-v0 \
    --episodes 10000 \
    --demos goto_redball_grey_10k
```

### 2. Train with Imitation Learning

```bash
uv run python scripts/train_il.py \
    --env BabyAI-GoToRedBallGrey-v0 \
    --arch unified_vit \
    --demos goto_redball_grey_10k \
    --demos-origin agent \
    --episodes 10000 \
    --model my_model \
    --epochs 100 \
    --val-interval 10
```

### 3. Fine-tune with PPO

IL→PPO is a well-established approach:
- **Sample efficiency**: IL gives a strong policy quickly; PPO from scratch on sparse rewards can take millions of frames to stumble on success
- **Avoiding local minima**: Random initialization often gets stuck; starting from a competent policy means PPO optimizes in a good region
- **Complementary strengths**: IL copies behavior but can't improve beyond demos; PPO optimizes for reward but needs good exploration. Combined: IL provides exploration, PPO provides optimization

```bash
uv run python scripts/train_rl.py \
    --env BabyAI-GoToRedBallGrey-v0 \
    --arch unified_vit \
    --pretrained-model my_model \
    --model my_model_ppo \
    --frames 1000000
```

### 4. Evaluate

```bash
uv run python scripts/evaluate.py \
    --env BabyAI-GoToRedBallGrey-v0 \
    --model my_model \
    --episodes 100
```

## Project Structure

```
toddler-ai/
├── src/toddler_ai/
│   ├── models/
│   │   └── unified_vit_model.py    # Main architecture
│   ├── algorithms/
│   │   ├── imitation.py            # Behavioral cloning
│   │   └── ppo.py                  # PPO RL
│   └── utils/
│       ├── format.py               # MiniLMPreprocessor with bert-tiny
│       └── agent.py                # ModelAgent for inference
├── scripts/
│   ├── train_il.py                 # IL training
│   ├── train_rl.py                 # RL training
│   ├── make_demos.py               # Demo generation
│   └── evaluate.py                 # Evaluation
└── demos/                          # Saved demonstrations
```

## Hardware

Automatically uses best available device:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon) - ~2500 FPS for IL training
- CPU fallback

## Memory System

The unified_vit uses episodic memory to store past experiences:
- 64 slots × 256-dim = 16,384 floats per batch
- Each memory entry is a full 256-dim pooled representation (goal + vision + context)
- Updated externally with shift-left-and-add pattern
- Temporal position encodings for smooth decay
- Reset to zeros at episode boundaries
- Self-attention learns which past experiences to retrieve

**Memory Update Pattern:**
```python
# In training and inference for unified_vit
embed_dim = model.embed_dim  # 256
new_memory = torch.zeros_like(memory)
new_memory[:, :-embed_dim] = memory[:, embed_dim:]  # Shift left by one entry
new_memory[:, -embed_dim:] = model_results['new_memory']  # Add new representation
memory = new_memory
```

## License

MIT License

## Acknowledgments

- [Farama Foundation](https://farama.org/) for Minigrid
- [Mila](https://mila.quebec/) for BabyAI
- OpenAI for PPO algorithm
