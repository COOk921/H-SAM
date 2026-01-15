"""
Core module for PPO training.

This module provides the core components for PPO training including
configuration management, trainer, evaluator, and metrics.
"""

from core.config import TrainingConfig, parse_args
from core.trainer import PPOTrainer
from core.evaluator import Evaluator
from core.metrics import (
    evaluate_correlation_metrics,
    compute_correlation_metrics,
)

__all__ = [
    'TrainingConfig',
    'parse_args',
    'PPOTrainer',
    'Evaluator',
    'evaluate_correlation_metrics',
    'compute_correlation_metrics',
]
