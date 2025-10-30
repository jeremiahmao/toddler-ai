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

### Models & Algorithms (from BabyAI)
- ✅ **FiLM-based Actor-Critic** - Language-conditioned vision model with GRU memory
- ✅ **PPO Algorithm** - Proximal Policy Optimization for RL training
- ✅ **Imitation Learning** - Complete IL training loop with demo support
- ✅ **Rule-based Bot** - Expert agent for generating demonstrations

### Training & Evaluation
- ✅ 6 ready-to-use scripts: train IL/RL, generate demos, evaluate, visualize
- ✅ TensorBoard logging and checkpointing
- ✅ Multi-environment training support
- ✅ Success rate tracking and data efficiency metrics

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
- tensorboardX → **torch.utils.tensorboard**

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

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,training]"
```

### Using pip

```bash
git clone https://github.com/jeremiahmao/toddler-ai.git
cd toddler-ai
pip install -e ".[dev,training]"
```

## Quick Start

### 1. Generate Demonstrations (using the bot)

```bash
python scripts/make_demos.py --env BabyAI-GoToLocal-v0 --episodes 1000 --demos demos/goto_local
```

The bot is a rule-based expert that can solve all BabyAI tasks perfectly, allowing you to generate training data without human demonstrations.

### 2. Train with Imitation Learning

```bash
# Small levels (GoToRedBall, GoToLocal, PickupLoc, PutNextLocal)
python scripts/train_il.py --env BabyAI-GoToLocal-v0 --demos goto_local \
    --batch-size 256 --val-episodes 512 --epoch-length 25600

# Larger levels (most other environments)
python scripts/train_il.py --env BabyAI-GoToDoor-v0 --demos goto_door \
    --memory-dim 2048 --recurrence 80 --batch-size 128 \
    --instr-arch attgru --instr-dim 256 --epoch-length 51200 --lr 5e-5
```

### 3. Train with Reinforcement Learning (PPO)

```bash
python scripts/train_rl.py --env BabyAI-GoToLocal-v0
```

Training typically takes several hours. Models and logs are saved to `models/` and `logs/` directories.

### 4. Evaluate Agent Performance

```bash
python scripts/evaluate.py --env BabyAI-GoToLocal-v0 --model <MODEL_NAME>
```

Evaluates on 1000 episodes and reports success rate.

### 5. Visualize Agent Behavior

```bash
python scripts/enjoy.py --env BabyAI-GoToLocal-v0 --model <MODEL_NAME>
```

Watch your trained agent solve tasks in real-time with rendering.

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
│   │   ├── imitation.py         # Imitation learning loop
│   │   ├── ppo.py               # PPO implementation
│   │   └── base.py              # Base RL algorithm
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
# Install all dependencies including dev tools
uv pip install -e ".[dev,training]"

# Install pre-commit hooks (auto-formats on commit)
pre-commit install
```

### Run Tests

```bash
pytest tests/
```

### Code Quality

This project uses pre-commit hooks to maintain code quality:
- **black** - Code formatting (line length: 100)
- **isort** - Import sorting
- **flake8** - Linting
- **pyright** - Type checking (basic mode)

```bash
# Run manually
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/

# Or just commit and hooks run automatically
git commit -m "your message"
```

## Repository Stats

- **46 Python modules** in `src/toddler_ai/`
- **6 training/evaluation scripts** ready to use
- **7 BabyAI environment types** (goto, open, pickup, putnext, synth, unlock, other)
- **3 core algorithms** (Imitation Learning, PPO, Base RL)
- **1 expert bot** for generating perfect demonstrations
- **100% modern Python** (3.10+, type hints, from `__future__` imports)

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
