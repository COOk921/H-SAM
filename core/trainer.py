"""
PPO Trainer for Operations Research Problems.

This module contains the PPOTrainer class that handles training loop,
rollout collection, advantage computation, and policy updates.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List, Any, Optional, Generator
from torch.utils.tensorboard import SummaryWriter

from core.config import TrainingConfig


class PPOTrainer:
    """PPO Trainer for vectorized environments."""
    
    def __init__(
        self,
        config: TrainingConfig,
        agent: nn.Module,
        envs,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        logger=None
    ):
        """
        Initialize PPO Trainer.
        
        Args:
            config: Training configuration
            agent: The RL agent
            envs: Vectorized training environments
            device: Torch device
            optimizer: Optimizer for the agent
            writer: TensorBoard writer
            logger: Logger instance
        """
        self.config = config
        self.agent = agent
        self.envs = envs
        self.device = device
        self.optimizer = optimizer
        self.writer = writer
        self.logger = logger
        
        self.global_step = 0
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize rollout storage buffers."""
        self.obs = [None] * self.config.num_steps
        self.actions = torch.zeros(
            (self.config.num_steps, self.config.num_envs, self.config.n_traj)
        ).to(self.device)
        self.logprobs = torch.zeros(
            (self.config.num_steps, self.config.num_envs, self.config.n_traj)
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.config.num_steps, self.config.num_envs, self.config.n_traj)
        ).to(self.device)
        self.dones = torch.zeros(
            (self.config.num_steps, self.config.num_envs, self.config.n_traj)
        ).to(self.device)
        self.values = torch.zeros(
            (self.config.num_steps, self.config.num_envs, self.config.n_traj)
        ).to(self.device)
    
    def collect_rollout(self) -> Dict[str, np.ndarray]:
        """
        Collect a rollout from the environment.
        
        Returns:
            Dictionary containing episode statistics
        """
        next_obs = self.envs.reset()
        encoder_state = self.agent.backbone.encode(next_obs)
        next_done = torch.zeros(self.config.num_envs, self.config.n_traj).to(self.device)
        
        episode_returns = np.zeros(
            (self.config.num_envs, self.config.n_traj), dtype=np.float32
        )
        episode_lengths = np.zeros(self.config.num_envs, dtype=np.int32)
        
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value, _ = self.agent.get_action_and_value_cached(
                    next_obs, state=encoder_state
                )

                # # --- POMO: Force diverse starting nodes at step 0 ---
                # if step == 0 and self.config.n_traj > 1:
                #     # Create range [0, 1, ..., n_traj-1]
                #     # Ensure n_traj matches the available nodes or is configured correctly
                #     pomo_action = torch.arange(
                #         self.config.n_traj, device=self.device
                #     ).repeat(self.config.num_envs, 1)
                    
                #     action = pomo_action
                    
                #     # Re-evaluate log_prob for the forced actions
                #     # Note: We use a temporary forward pass to get log_prob of forced actions
                #     _, forced_logprob, _, _, _ = self.agent.get_action_and_value_cached(
                #         next_obs, action=action, state=encoder_state
                #     )
                #     logprob = forced_logprob
                # # ----------------------------------------------------
                
                action = action.view(self.config.num_envs, self.config.n_traj)
                self.values[step] = value.view(self.config.num_envs, self.config.n_traj)
            
            self.actions[step] = action
            self.logprobs[step] = logprob.view(self.config.num_envs, self.config.n_traj)
            
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device)
            next_done = torch.Tensor(done).to(self.device)
            
            episode_returns += reward
            episode_lengths += 1
        
        # Store final state for advantage computation
        self._next_obs = next_obs
        self._next_done = next_done
        self._encoder_state = encoder_state
        
        return {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths
        }
    
    def compute_advantages(self) -> torch.Tensor:
        """
        Compute GAE advantages and returns.
        
        Returns:
            Returns tensor
        """
        with torch.no_grad():
            next_value = self.agent.get_value_cached(
                self._next_obs, self._encoder_state
            ).squeeze(-1)
            
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = torch.zeros(
                self.config.num_envs, self.config.n_traj
            ).to(self.device)
            
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - self._next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                delta = (
                    self.rewards[t] 
                    + self.config.gamma * nextvalues * nextnonterminal 
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta 
                    + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                )
            
            returns = advantages + self.values
        
        self._advantages = advantages
        self._returns = returns
        
        return returns
    
    def update_policy(self, update: int) -> Dict[str, float]:
        """
        Update policy using collected rollout.
        
        Args:
            update: Current update iteration
            
        Returns:
            Dictionary of training metrics
        """
        # Flatten batch observations (excluding graph_data)
        b_obs = {
            k: np.concatenate([obs_[k] for obs_ in self.obs]) 
            for k in self.envs.single_observation_space 
            if k != "graph_data"
        }
        
        b_logprobs = self.logprobs.reshape(-1, self.config.n_traj)
        b_actions = self.actions.reshape(
            (-1,) + self.envs.single_action_space.shape
        )
        b_advantages = self._advantages.reshape(-1, self.config.n_traj)
        b_returns = self._returns.reshape(-1, self.config.n_traj)
        b_values = self.values.reshape(-1, self.config.n_traj)
        
        # Optimization setup
        envsperbatch = self.config.num_envs // self.config.num_minibatches
        envinds = np.arange(self.config.num_envs)
        flatinds = np.arange(self.config.batch_size).reshape(
            self.config.num_steps, self.config.num_envs
        )
        
        clipfracs = []
        
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(envinds)
            
            for start in range(0, self.config.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()
                r_inds = np.tile(np.arange(envsperbatch), self.config.num_steps)
                
                # Prepare current observation for encoding
                cur_obs = {}
                for k, v in self.obs[0].items():
                    if k == "data_key":
                        continue
                    if k == "graph_data":
                        graph_items = [v[i] for i in mbenvinds]
                        cur_obs[k] = graph_items
                    else:
                        cur_obs[k] = v[mbenvinds]
                
                encoder_state = self.agent.backbone.encode(cur_obs)
                
                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value_cached(
                    {k: v[mb_inds] for k, v in b_obs.items()},
                    b_actions.long()[mb_inds],
                    (embedding[r_inds, :] for embedding in encoder_state),
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()
                    )
                
                mb_advantages = b_advantages[mb_inds]
                if self.config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1, self.config.n_traj)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = (
                    pg_loss 
                    - self.config.ent_coef * entropy_loss 
                    + v_loss * self.config.vf_coef
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
            
            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break
        
        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return {
            "v_loss": v_loss.item(),
            "pg_loss": pg_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "explained_variance": explained_var,
        }
    
    def train_step(self, update: int) -> Dict[str, Any]:
        """
        Execute a complete training step.
        
        Args:
            update: Current update iteration
            
        Returns:
            Dictionary of all training metrics
        """
        self.agent.train()
        
        # Learning rate annealing
        if self.config.anneal_lr:
            frac = 1.0 - (update - 1.0) / self.config.num_updates
            lrnow = frac * self.config.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
        
        # Collect rollout
        episode_stats = self.collect_rollout()
        
        # Compute advantages
        self.compute_advantages()
        
        # Update policy
        update_metrics = self.update_policy(update)
        
        # Combine metrics
        avg_return = np.mean(np.mean(episode_stats["episode_returns"], axis=1))
        max_return = np.mean(np.max(episode_stats["episode_returns"], axis=1))
        avg_length = np.mean(episode_stats["episode_lengths"])
        
        metrics = {
            "avg_episodic_return": avg_return,
            "max_episodic_return": max_return,
            "avg_episodic_length": avg_length,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **update_metrics,
        }
        
        # Log metrics
        self._log_metrics(metrics, self.global_step)
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float], global_step: int):
        """Log metrics to TensorBoard and logger."""
        if self.writer is not None:
            self.writer.add_scalar(
                "charts/episodic_return_mean", 
                metrics["avg_episodic_return"], 
                global_step
            )
            self.writer.add_scalar(
                "charts/episodic_return_max", 
                metrics["max_episodic_return"], 
                global_step
            )
            self.writer.add_scalar(
                "charts/episodic_length", 
                metrics["avg_episodic_length"], 
                global_step
            )
            self.writer.add_scalar(
                "charts/learning_rate", 
                metrics["learning_rate"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/value_loss", 
                metrics["v_loss"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/policy_loss", 
                metrics["pg_loss"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/entropy", 
                metrics["entropy_loss"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/old_approx_kl", 
                metrics["old_approx_kl"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/approx_kl", 
                metrics["approx_kl"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/clipfrac", 
                metrics["clipfrac"], 
                global_step
            )
            self.writer.add_scalar(
                "losses/explained_variance", 
                metrics["explained_variance"], 
                global_step
            )
        
        if self.logger is not None:
            self.logger.info(
                f"[Train] global_step={global_step}\n"
                f"avg_episodic_return={metrics['avg_episodic_return']}\n"
                f"max_episodic_return={metrics['max_episodic_return']}\n"
                f"avg_episodic_length={metrics['avg_episodic_length']}"
            )
            self.logger.info("")
