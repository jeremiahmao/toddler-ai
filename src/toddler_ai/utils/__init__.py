"""Utility functions and helpers for Toddler AI."""

from __future__ import annotations

import os
import random
import numpy
import torch

from toddler_ai.utils.agent import load_agent, Agent, ModelAgent, BotAgent, DemoAgent
from toddler_ai.utils.demos import get_demos_path, load_demos, save_demos
from toddler_ai.utils.model import get_model_dir, get_model_path, load_model, save_model
from toddler_ai.utils.log import configure_logging, get_log_path, get_log_dir, synthesize
from toddler_ai.utils.format import get_vocab_path, ObssPreprocessor, IntObssPreprocessor


def storage_dir():
    """Get the storage directory for models, logs, and demos."""
    return os.environ.get("TODDLER_AI_STORAGE", '.')


def create_folders_if_necessary(path):
    """Create parent directories for a path if they don't exist."""
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def seed(seed_value):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


__all__ = [
    "seed",
    "load_agent",
    "Agent",
    "ModelAgent",
    "BotAgent",
    "DemoAgent",
    "get_demos_path",
    "load_demos",
    "save_demos",
    "get_model_dir",
    "get_model_path",
    "load_model",
    "save_model",
    "configure_logging",
    "get_log_path",
    "get_log_dir",
    "synthesize",
    "get_vocab_path",
    "storage_dir",
    "create_folders_if_necessary",
]
