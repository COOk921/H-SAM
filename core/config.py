"""
Configuration management for PPO training.

This module provides configuration classes and argument parsing for the RL training pipeline.
"""

import argparse
import os
from dataclasses import dataclass, field
from distutils.util import strtobool
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for PPO training."""
    
    # Experiment settings
    exp_name: str = "ppo_or"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    
    # Problem settings
    problem: str = "cvrp"
    env_id: str = "cvrp-v0"
    env_entry_point: str = "envs.cvrp_vector_env:CVRPVectorEnv"
    
    # Training hyperparameters
    total_timesteps: int = 15_000_000
    learning_rate: float = 5e-4
    weight_decay: float = 0
    num_envs: int = 1024
    num_steps: int = 100
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 2
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    
    # POMO settings
    n_traj: int = 50
    n_test: int = 100
    multi_greedy_inference: bool = True
    
    # Computed properties (set in __post_init__)
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    num_updates: int = field(init=False)
    
    def __post_init__(self):
        """Compute derived values after initialization."""
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // self.batch_size


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig object."""
    parser = argparse.ArgumentParser(description="PPO for Operations Research Problems")
    
    # Experiment settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), 
        default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), 
        default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances")
    
    # Problem settings
    parser.add_argument("--problem", type=str, default="cvrp",
        help="the OR problem we are trying to solve")
    parser.add_argument("--env-id", type=str, default="cvrp-v0",
        help="the id of the environment")
    parser.add_argument("--env-entry-point", type=str, 
        default="envs.cvrp_vector_env:CVRPVectorEnv",
        help="the path to the environment class definition")
    
    # Training hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=0,
        help="the weight decay of the optimizer")
    parser.add_argument("--num-envs", type=int, default=512,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=100,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="Toggle learning rate annealing")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for GAE")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="Toggles clipped value loss")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # POMO settings
    parser.add_argument("--n-traj", type=int, default=50,
        help="number of trajectories in a vectorized sub-environment")
    parser.add_argument("--n-test", type=int, default=100,
        help="how many test instances")
    parser.add_argument("--multi-greedy-inference", type=lambda x: bool(strtobool(x)), 
        default=True, nargs="?", const=True,
        help="whether to use multiple trajectory greedy inference")
    
    args = parser.parse_args()
    
    # Convert to TrainingConfig
    config = TrainingConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        cuda=args.cuda,
        track=args.track,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        capture_video=args.capture_video,
        problem=args.problem,
        env_id=args.env_id,
        env_entry_point=args.env_entry_point,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        n_traj=args.n_traj,
        n_test=args.n_test,
        multi_greedy_inference=args.multi_greedy_inference,
    )
    
    return config
