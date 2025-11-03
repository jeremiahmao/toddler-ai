"""Training algorithms (RL and IL) for Toddler AI."""

from __future__ import annotations

from toddler_ai.algorithms.ppo import PPOAlgo
from toddler_ai.algorithms.imitation import ImitationLearning

__all__ = ["PPOAlgo", "ImitationLearning"]
