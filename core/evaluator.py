"""
Evaluator for RL models.

This module contains the Evaluator class for evaluating trained models
on test environments.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from core.config import TrainingConfig
from core.metrics import compute_correlation_metrics


class Evaluator:
    """Evaluator for RL models on test environments."""
    
    def __init__(
        self,
        config: TrainingConfig,
        agent: nn.Module,
        test_envs,
        device: torch.device,
        writer: Optional[SummaryWriter] = None,
        logger=None
    ):
        """
        Initialize Evaluator.
        
        Args:
            config: Training configuration
            agent: The RL agent
            test_envs: Vectorized test environments
            device: Torch device
            writer: TensorBoard writer
            logger: Logger instance
        """
        self.config = config
        self.agent = agent
        self.test_envs = test_envs
        self.device = device
        self.writer = writer
        self.logger = logger
    
    def evaluate(self, global_step: int) -> Dict[str, float]:
        """
        Run evaluation on test environments.
        
        Args:
            global_step: Current global step for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.agent.eval()
        test_obs = self.test_envs.reset()
        
        trajectories = []
        episode_returns = 0
        episode_lengths = 0
        
        for step in range(self.config.num_steps):
            with torch.no_grad():
                action, logits = self.agent(test_obs)
            
            # Multi-greedy inference for first step
            if step == 0 and self.config.multi_greedy_inference:
                action = self._get_initial_action(action)
            
            test_obs, reward, _, test_info = self.test_envs.step(action.cpu().numpy())
            trajectories.append(action.cpu().numpy())
            
            episode_returns += reward
            episode_lengths += 1
        
        # Process trajectories and compute metrics
        metrics = self._compute_metrics(trajectories, episode_returns)
        
        # Log metrics
        self._log_metrics(metrics, global_step)
        
        return metrics
    
    def _get_initial_action(self, action: torch.Tensor) -> torch.Tensor:
        """Get initial action for multi-greedy inference."""
        if self.config.problem == 'container':
            return torch.arange(self.config.n_traj).repeat(self.config.n_test, 1)
        elif self.config.problem == 'tsp':
            return torch.arange(self.config.n_traj).repeat(self.config.n_test, 1)
        elif self.config.problem == 'cvrp':
            return torch.arange(1, self.config.n_traj + 1).repeat(self.config.n_test, 1)
        return action
    
    def _compute_metrics(
        self, 
        trajectories: list, 
        episode_returns: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics from trajectories."""
        # Shape: (env, traj, step)
        resulting_traj = np.array(trajectories).transpose(1, 2, 0)
        
        # Compute target sequence
        target = np.concatenate([
            np.arange(resulting_traj.shape[-1] - 1), [0]
        ])
        target = np.tile(
            target, 
            (resulting_traj.shape[0], resulting_traj.shape[1], 1)
        )
        
        # Compute correlation metrics
        tau_mean, rho_mean, tau_max, rho_max = compute_correlation_metrics(
            resulting_traj.copy(), target
        )
        
        # Compute return metrics
        avg_return = np.mean(np.mean(episode_returns, axis=1))
        max_return = np.mean(np.max(episode_returns, axis=1))
        
        return {
            "avg_episodic_return": avg_return,
            "max_episodic_return": max_return,
            "tau_mean": tau_mean,
            "rho_mean": rho_mean,
            "tau_max": tau_max,
            "rho_max": rho_max,
        }
    
    def _log_metrics(self, metrics: Dict[str, float], global_step: int):
        """Log evaluation metrics."""
        if self.writer is not None:
            self.writer.add_scalar(
                "test/episodic_return_mean", 
                metrics["avg_episodic_return"], 
                global_step
            )
            self.writer.add_scalar(
                "test/tau_mean", 
                metrics["tau_mean"], 
                global_step
            )
            self.writer.add_scalar(
                "test/rho_mean", 
                metrics["rho_mean"], 
                global_step
            )
            self.writer.add_scalar(
                "test/tau_max", 
                metrics["tau_max"], 
                global_step
            )
            self.writer.add_scalar(
                "test/rho_max", 
                metrics["rho_max"], 
                global_step
            )
        
        if self.logger is not None:
            self.logger.info(
                "--------------------------------------------"
                f"[test] episodic_return={metrics['max_episodic_return']}\n"
                f"avg_episodic_return={metrics['avg_episodic_return']}\n"
                f"tau_mean={metrics['tau_mean']}\n"
                f"rho_mean={metrics['rho_mean']}\n"
                f"tau_max={metrics['tau_max']}\n"
                f"rho_max={metrics['rho_max']}\n"
                "--------------------------------------------"
            )
            self.logger.info("")
