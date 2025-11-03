# Toddler AI

A modern, clean implementation combining Minigrid environments with BabyAI training infrastructure for grounded language learning research.

## Overview

Toddler AI integrates the best components from two foundational projects:

- **Environments** from [Minigrid](https://github.com/Farama-Foundation/Minigrid) (Farama Foundation) - BabyAI grid-world environments with language instructions
- **Training Infrastructure** from [BabyAI](https://github.com/mila-iqia/babyai) (Mila) - Complete imitation learning and RL training pipelines
- **Modern Tooling** - Modernized with Python 3.10+, PyTorch 2.0+, Gymnasium, and `uv` package management

## What's Included

This repository contains a curated, organized selection of production-ready code:

### Environments (from Minigrid)
- ✅ Core Minigrid engine (`minigrid_env.py`, grid system, actions, objects)
- ✅ 7 BabyAI environment categories (GoTo, Open, Pickup, PutNext, Synth, Unlock, Other)
- ✅ Environment wrappers for observation/action space customization
- ✅ Level generation and verification utilities

### Models & Algorithms
- ✅ **Vision Transformer (ViT)** - Modern attention-based architecture for vision-language grounding
- ✅ **MiniLM Integration** - Pretrained language encoder (384-dim) with differential learning rates
- ✅ **PPO Algorithm** - Proximal Policy Optimization ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) with advantage normalization and optimized defaults for sparse rewards
- ✅ **Imitation Learning** - Behavioral cloning for training from demonstrations (from BabyAI)
- ✅ **Rule-based Bot** - Expert agent for generating demonstrations (from BabyAI)
- ⚠️ **FiLM-based Actor-Critic** - DEPRECATED: Legacy CNN architecture (from BabyAI, kept for compatibility)

### Training & Evaluation
- ✅ 6 ready-to-use scripts: train IL/RL, generate demos, evaluate, visualize
- ✅ Weights & Biases (wandb) integration for experiment tracking
- ✅ Multi-environment training support
- ✅ Success rate tracking and data efficiency metrics
- ✅ Automatic GPU acceleration (CUDA on NVIDIA, CPU fallback for Apple Silicon)

### Development Tools
- ✅ Modern `pyproject.toml` with `uv` support (replaces old `setup.py`)
- ✅ Pre-commit hooks (black, isort, flake8)
- ✅ pytest test suite
- ✅ Type checking with pyright

## Key Updates from Original Repos

**Modernized Dependencies:**
- Python 3.6 → **3.10+**
- PyTorch 0.4.1 → **2.0+**
- gym → **gymnasium 0.29+**
- tensorboardX → **Weights & Biases (wandb)**
- Automatic GPU acceleration (CUDA supported, MPS in progress)

**Improved Organization:**
- Flat structure → **clean `src/` layout**
- Mixed modules → **separated envs/models/algorithms/agents/utils**
- Old packaging → **modern pyproject.toml with uv**

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/jeremiahmao/toddler-ai.git
cd toddler-ai

# Install core dependencies (creates venv automatically)
uv sync

# Optional: Install wandb for experiment tracking
uv sync --extra tracking
```

### Using pip

```bash
git clone https://github.com/jeremiahmao/toddler-ai.git
cd toddler-ai
pip install -e .

# Optional: Install wandb for experiment tracking
pip install -e ".[tracking]"
```

## Quick Start

**By default, Toddler AI uses the modern ViT + MiniLM architecture.** All training scripts automatically use Vision Transformer for vision and pretrained MiniLM for language understanding. Legacy FiLM-based architectures are deprecated.

### 1. Generate Demonstrations (using the bot)

```bash
uv run python scripts/make_demos.py --env BabyAI-GoToLocal-v0 --episodes 10 --valid-episodes 5 --demos demos/goto_local
```

The bot is a rule-based expert that can solve all BabyAI tasks perfectly, allowing you to generate training data without human demonstrations.

### 2. Train with Imitation Learning

**ViT + MiniLM is now the default** - no need to specify `--arch` or `--instr-arch`!

```bash
# Quick test run (uses ViT + MiniLM by default)
uv run python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos demos/goto_local \
    --model test_model --batch-size 10 --epochs 50 --val-interval 10

# With Weights & Biases tracking (requires: uv sync --extra tracking)
uv run python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos demos/goto_local \
    --model test_model --tb

# Small levels (GoToRedBall, GoToLocal, PickupLoc, PutNextLocal)
uv run python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos goto_local \
    --batch-size 256 --val-episodes 512

# Larger levels
uv run python scripts/train_il.py --env BabyAI-GoToDoor-v0 --demos goto_door \
    --memory-dim 2048 --recurrence 80 --batch-size 128 --epoch-length 51200
```

**Legacy FiLM architecture** (deprecated, for compatibility only):
```bash
uv run python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos demos/goto_local \
    --arch bow_endpool_res --instr-arch gru --model legacy_model
```

### 3. Train with Reinforcement Learning (PPO)

**PPO (Proximal Policy Optimization)** is the primary RL algorithm in Toddler AI. It uses clipped surrogate objectives and value function clipping for stable, efficient policy learning. **ViT + MiniLM is now the default architecture.**

```bash
# Basic PPO training (uses ViT + MiniLM by default)
uv run python scripts/train_rl.py --env BabyAI-GoToLocal-v0

# PPO with custom hyperparameters
uv run python scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
    --frames 1000000 --lr 1e-4 --clip-eps 0.2 --ppo-epochs 4 \
    --batch-size 256 --frames-per-proc 128 --discount 0.99 --gae-lambda 0.99

# PPO with pretrained model from imitation learning
uv run python scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
    --pretrained-model models/your_il_model
```

**Legacy FiLM architecture** (deprecated):
```bash
uv run python scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
    --arch bow_endpool_res --instr-arch gru
```

**Key PPO hyperparameters (optimized for sparse rewards):**
- `--reward-scale`: Reward multiplier (default: **1.0**, optimal for BabyAI's sparse binary rewards)
- `--value-loss-coef`: Value loss coefficient (default: **0.25**, tuned for stability)
- `--clip-eps`: PPO clipping parameter (default: 0.2)
- `--ppo-epochs`: Number of PPO update epochs per batch (default: 4)
- `--batch-size`: Batch size for PPO updates (default: 256)
- `--gae-lambda`: GAE lambda for advantage estimation (default: 0.99)
- `--discount`: Reward discount factor (default: 0.99)

**⚡ Stability Improvements:** PPO now includes automatic advantage normalization and optimized defaults for sparse reward environments. Training is stable out-of-the-box with ~400 FPS on CPU and smooth convergence.

Training typically takes several hours. Models and logs are saved to `models/` and `logs/` directories.

**Experiment Tracking with Weights & Biases:**

Add the `--tb` flag to any training command to log metrics to [wandb.ai](https://wandb.ai):
- Beautiful interactive dashboards
- Compare multiple runs
- Track hyperparameters automatically
- Free for personal/academic use

First time setup:
```bash
uv sync --extra tracking  # Install wandb
uv run wandb login        # Login with your wandb account
```

Then train with tracking:
```bash
uv run python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos demos/goto_local --model my_model --tb
```

View your experiments at: https://wandb.ai

## Hardware Acceleration

Toddler AI automatically detects and uses the best available device:

**NVIDIA GPUs (CUDA):**
- Fully supported for both IL and RL training
- 3-5x speedup over CPU
- Automatically detected and used

**Apple Silicon (MPS):**
- Currently disabled due to PyTorch MPS limitations with certain operations
- Training runs on CPU (still fast for these small models)
- Will be enabled once PyTorch MPS support matures
- MPS will provide significant benefits when we integrate larger language models (MiniLM coming soon)

**CPU:**
- Works well for all training tasks
- ~60 FPS for PPO training on modern CPUs
- Default fallback if no GPU available

The code automatically selects the best device with priority: CUDA > MPS > CPU. No configuration needed.

### 4. Evaluate Agent Performance

```bash
uv run python scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model test_model --episodes 10 --argmax
```

Evaluates on specified number of episodes and reports success rate.

### 5. Visualize Agent Behavior

```bash
uv run python scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model test_model
```

Watch your trained agent solve tasks in real-time with rendering.

## Model Architectures

### Vision Transformer (ViT) - RECOMMENDED ✨

**Modern attention-based architecture (default):**
```
Image → Patch Embeddings → Self-Attention → Cross-Attention ← MiniLM
                                                    ↓
                                            Pool → Memory → Actor/Critic
```

**Architecture Flow:**
1. **Patch Embedding (271K params)**: 7×7 image → 49 patch tokens (128-dim each)
2. **Vision Self-Attention (271K params)**: Patches reason about spatial relationships
3. **Cross-Attention (271K params)**: Vision queries language for grounding
4. **Mean Pooling**: 49 patches → single 128-dim vector
5. **LSTM Memory** (optional): Temporal context
6. **Actor/Critic Heads**: Action distribution + value function

**With MiniLM (23.2M trainable params):**
```
Text → MiniLM (22.7M) → Projection (49K) → Cross-Attn ← Vision
                                                ↓
                                        Pool → Memory → Actor/Critic
```

**Why ViT?** Modern attention-based architecture that works across modalities (spatial, temporal, multi-modal). ViT is efficient for BabyAI's 7×7 grids (only 49 patches!) and provides better generalization than CNNs.

## MiniLM Language Encoder

ViT uses pretrained MiniLM (all-MiniLM-L6-v2, 384-dim, 22.7M params) as the default language encoder, providing better language understanding than training a GRU from scratch.

### Three Training Modes

#### 1. FREEZE Mode (Fast, 1.25M trainable params)
```bash
uv run python scripts/train_il.py \
    --env BabyAI-GoToLocal-v0 \
    --demos demos/goto_local \
    --instr-arch minilm \
    --freeze-minilm
```
- MiniLM: **FROZEN** ❄️ (0 params trained)
- Only projection + model train
- **Use when:** Maximum speed, trust pretrained knowledge

#### 2. DECAY Mode - RECOMMENDED (23.2M trainable params with ViT)
```bash
uv run python scripts/train_il.py \
    --env BabyAI-GoToLocal-v0 \
    --demos demos/goto_local \
    --arch vit \
    --instr-arch minilm
    # This is the default - no extra flags needed!
```

**3-Tier Differential Learning Rates (ViT):**
- MiniLM: **HIGH INERTIA** 🏋️ (22.7M params @ LR=1e-6, weight_decay=0.1)
- ViT components: **MEDIUM INERTIA** ⚖️ (271K params @ LR=1e-5, weight_decay=0.01)
- Task-specific: **TRAINS FREELY** 🆓 (215K params @ LR=1e-4, weight_decay=0.0)

**Use when:** Want adaptation while protecting pretrained knowledge (DEFAULT for ViT + MiniLM)

**What "high inertia" means:**
```python
Loss = Task_Loss + 0.1 * ||θ_minilm - θ_pretrained||²
```
MiniLM resists moving away from pretrained weights via L2 regularization.

#### 3. FREE Mode (Maximum adaptation, 23.9M trainable params)
```bash
uv run python scripts/train_il.py \
    --env BabyAI-GoToLocal-v0 \
    --demos demos/goto_local \
    --instr-arch minilm \
    --minilm-lr-multiplier 1.0 \
    --minilm-weight-decay 0.0
```
- MiniLM + Model: **BOTH TRAIN FREELY** 🆓
- **Use when:** Lots of data, want maximum task adaptation
- **Risk:** Catastrophic forgetting of pretrained knowledge

### Fine-tuning Controls

```bash
--minilm-lr-multiplier 0.01       # LR multiplier for encoder (0.01 = 100x smaller)
--minilm-weight-decay 0.1         # Inertia strength (higher = more resistance)
--freeze-minilm                   # Completely freeze encoder
```

### Installation

```bash
# Install language model support
uv sync --extra language

# Verify installation
uv run python -c "from sentence_transformers import SentenceTransformer; print('✓ Ready!')"
```

### Comparison: GRU vs MiniLM

| Feature | GRU (default) | MiniLM |
|---------|---------------|---------|
| Params | ~130K | 22.7M (freeze) or 23.9M (finetune) |
| Pretrained | ❌ | ✅ |
| Language understanding | Task-specific | General + task-adapted |
| Training time | Fast | Slower (more params) |
| Best for | Small envs, fast experiments | Complex instructions, transfer learning |

### FiLM-based CNN (DEPRECATED) ⚠️

**Legacy architecture kept for compatibility only.** Use ViT for new projects.

**Feature-wise Linear Modulation - vision-specific conditioning:**
```
vision_features = conv(image)
γ = weight(language_embedding)  # Scale
β = bias(language_embedding)    # Shift
output = γ * vision_features + β  # Language conditions vision!
```

**With MiniLM:**
```
Text → MiniLM (22.7M) → Projection (49K) → FiLM → Actor/Critic
```

**When to use FiLM:**
- You have existing FiLM-trained models to fine-tune
- You need compatibility with older BabyAI code
- **For new projects, use ViT instead**

**Usage:**
```bash
# Legacy FiLM architecture
uv run python scripts/train_rl.py --env BabyAI-GoToLocal-v0 \
    --arch bow_endpool_res --instr-arch gru
```

## PPO Training Stability

Toddler AI's PPO implementation has been optimized for **sparse reward environments** like BabyAI, where rewards are typically binary (0 for failure, 1 for success).

### Critical Improvements

**1. Automatic Advantage Normalization**

PPO now normalizes advantages before computing policy loss, preventing gradient explosions with reward scaling:

```python
# Normalize advantages per mini-batch (critical for stability)
advantage_normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
```

Without this, sparse rewards + reward scaling cause advantages to have std ~8.5 instead of ~1.0, leading to:
- Exploding policy gradients (grad norm > 18)
- Value loss spikes (up to 50x normal)
- Volatile success rates (0% → 75% → 0%)

**2. Optimized Defaults for Sparse Rewards**

| Parameter | Old Default | New Default | Why Changed |
|-----------|-------------|-------------|-------------|
| `reward_scale` | 20.0 | **1.0** | Prevents value loss explosion (380 → 0.25) |
| `value_loss_coef` | 0.5 | **0.25** | Balances policy/value learning |

**Why this matters:** With `reward_scale=20` and binary rewards [0, 1], returns become [0, 20]. If the value function predicts 0.5 for a return of 20, the squared error is `(20 - 0.5)² = 380.25`, which **dominates** the total loss and causes training instability.

### Performance Comparison

| Metric | Before (20.0, 0.5) | After (1.0, 0.25) |
|--------|-------------------|------------------|
| **Value Loss** | 1.0 → 52.8 ⚠️ | 0.02 → 0.04 ✅ |
| **Gradient Norm** | Up to 18.9 ⚠️ | 0.1 → 0.3 ✅ |
| **Success Rate** | 0% → 75% → 0% ⚠️ | 24% → 46% ✅ |
| **FPS** | ~200 | ~400 |
| **Stability** | Volatile | Smooth convergence |

### When to Override Defaults

The new defaults work best for BabyAI's sparse binary rewards. Override if:
- Your environment has **dense rewards** (reward every step)
- Your environment has **continuous rewards** (not just 0/1)
- Rewards are already **normalized** to a small range

```bash
# For dense/continuous reward environments
uv run python scripts/train_rl.py --env YourEnv-v0 \
    --reward-scale 10.0 --value-loss-coef 0.5
```

## Project Structure

```
toddler-ai/
├── src/toddler_ai/              # Main package (46 Python files)
│   ├── envs/                    # Environment definitions
│   │   ├── core/                # Core Minigrid components (grid, actions, objects)
│   │   ├── babyai/              # BabyAI language-conditioned environments
│   │   │   └── core/            # Level generation and verification
│   │   ├── minigrid_env.py      # Base Minigrid environment class
│   │   └── wrappers.py          # Observation/action wrappers
│   ├── models/                  # Neural network models
│   │   ├── ac_model.py          # Actor-Critic with FiLM conditioning
│   │   ├── rl_base.py           # Base RL model interface
│   │   └── format.py            # Model formatting utilities
│   ├── algorithms/              # Training algorithms
│   │   ├── ppo.py               # PPO (Proximal Policy Optimization) - Primary RL method
│   │   ├── imitation.py         # Imitation learning from demonstrations
│   │   └── base.py              # Base RL algorithm interface
│   ├── agents/                  # Agent implementations
│   │   └── bot.py               # Rule-based expert bot
│   └── utils/                   # Utilities (12 modules)
│       ├── demos.py             # Demo loading/saving
│       ├── agent.py             # Agent utilities
│       ├── model.py             # Model save/load
│       ├── log.py               # Logging
│       └── ...                  # And more
├── scripts/                     # Executable scripts (6 files)
│   ├── train_il.py              # Train with imitation learning
│   ├── train_rl.py              # Train with RL (PPO)
│   ├── make_demos.py            # Generate demonstrations
│   ├── evaluate.py              # Evaluate model success rate
│   ├── enjoy.py                 # Visualize agent behavior
│   └── manual_control.py        # Human control interface
├── tests/                       # Unit tests
├── pyproject.toml               # Modern Python packaging (uv-compatible)
├── .pre-commit-config.yaml      # Code quality hooks
└── README.md                    # You are here
```

## Development

### Setup Development Environment

```bash
# Install all dependencies (uv sync installs everything from pyproject.toml)
uv sync

# Install pre-commit hooks (auto-formats on commit)
uv run pre-commit install
```

### Run Tests

```bash
uv run pytest tests/
```

### Code Quality

This project uses pre-commit hooks to maintain code quality:
- **black** - Code formatting (line length: 100)
- **isort** - Import sorting
- **flake8** - Linting
- **pyright** - Type checking (basic mode)

```bash
# Run manually
uv run black src/ scripts/ tests/
uv run isort src/ scripts/ tests/
uv run flake8 src/ scripts/ tests/

# Or just commit and hooks run automatically
git commit -m "your message"
```

## Repository Stats

- **46 Python modules** in `src/toddler_ai/`
- **6 training/evaluation scripts** ready to use
- **7 BabyAI environment types** (goto, open, pickup, putnext, synth, unlock, other)
- **2 training algorithms** (PPO for RL, Imitation Learning for behavioral cloning)
- **1 expert bot** for generating perfect demonstrations
- **100% gymnasium API** (modern replacement for OpenAI Gym)
- **Automatic GPU detection** (CUDA > MPS > CPU priority)
- **Modern experiment tracking** (Weights & Biases integration)

## Citation

This project builds upon:

**Minigrid:**
```bibtex
@inproceedings{MinigridMiniworld23,
  author       = {Maxime Chevalier{-}Boisvert and Bolun Dai and Mark Towers and Rodrigo Perez{-}Vicente and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid {\&} Miniworld: Modular {\&} Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  booktitle    = {NeurIPS},
  year         = {2023},
}
```

**BabyAI:**
```bibtex
@inproceedings{babyai_iclr19,
  title={BabyAI: First Steps Towards Grounded Language Learning With a Human In the Loop},
  author={Maxime Chevalier-Boisvert and Dzmitry Bahdanau and Salem Lahlou and Lucas Willems and Chitwan Saharia and Thien Huu Nguyen and Yoshua Bengio},
  booktitle={ICLR},
  year={2019},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Farama Foundation](https://farama.org/) for maintaining Minigrid
- [Mila](https://mila.quebec/en/) for creating BabyAI
- Original authors of both projects
