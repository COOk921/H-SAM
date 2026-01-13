"""
PPO Training for Operations Research Problems.

This is the main entry point for training PPO agents on various OR problems
including Container, TSP, and CVRP.

Usage:
    python ppo_or.py --num-steps 51 --env-id container-v0 \\
        --env-entry-point envs.container_vector_env:ContainerVectorEnv \\
        --problem container
"""

import logging
import os
import random
import shutil
import time
import warnings

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.config import parse_args
from core.trainer import PPOTrainer
from core.evaluator import Evaluator
from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv

warnings.filterwarnings("ignore")


def make_env(env_id: str, seed: int, cfg: dict = None):
    """Create environment factory function."""
    if cfg is None:
        cfg = {}
    
    def thunk():
        env = gym.make(env_id, **cfg)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk


def set_seed(seed: int, torch_deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def setup_experiment(config, run_name: str):
    """Setup experiment directory and logging."""
    # Initialize WandB if tracking
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config) if hasattr(config, '__dict__') else config.__dict__,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    # Setup TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in 
                      (vars(config) if hasattr(config, '__dict__') else config.__dict__).items()])),
    )
    
    # Create checkpoint directory
    os.makedirs(os.path.join(f"runs/{run_name}", "ckpt"), exist_ok=True)
    
    # Copy main script for reference
    shutil.copy(__file__, os.path.join(f"runs/{run_name}", "main.py"))
    
    # Setup logging
    logging.basicConfig(
        filename=f'runs/{run_name}/training.log',
        level=logging.INFO,
        format='%(message)s'
    )
    
    return writer


def create_train_envs(config):
    """Create training environments."""
    return SyncVectorEnv([
        make_env(config.env_id, config.seed + i, dict(mode="train"))
        for i in range(config.num_envs)
    ])


def create_test_envs(config):
    """Create test environments."""
    return SyncVectorEnv([
        make_env(
            config.env_id,
            config.seed + config.num_envs + i,
            dict(mode="test", index=i)
        )
        for i in range(config.n_test)
    ])


def save_checkpoint(agent, run_name: str, update: int):
    """Save model checkpoint."""
    torch.save(
        agent.state_dict(),
        f"runs/{run_name}/ckpt/{update}.pt"
    )


def main():
    """Main training function."""
    # Parse configuration
    config = parse_args()
    
    # Setup run name
    run_name = f"{config.env_id}__{config.exp_name}__{time.strftime('%Y-%m-%d_%H_%M')}"
    
    # Setup experiment
    writer = setup_experiment(config, run_name)
    
    # Set random seeds
    set_seed(config.seed, config.torch_deterministic)
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
    )
    
    # Register environment
    gym.envs.register(
        id=config.env_id,
        entry_point=config.env_entry_point,
    )
    
    # Create environments
    train_envs = create_train_envs(config)
    test_envs = create_test_envs(config)
    
    assert isinstance(
        train_envs.single_action_space, gym.spaces.MultiDiscrete
    ), "only discrete action space is supported"
    
    # Create agent
    agent = Agent(device=device, name=config.problem).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
        weight_decay=config.weight_decay
    )
    
    # Create trainer and evaluator
    trainer = PPOTrainer(
        config=config,
        agent=agent,
        envs=train_envs,
        device=device,
        optimizer=optimizer,
        writer=writer,
        logger=logging
    )
    
    evaluator = Evaluator(
        config=config,
        agent=agent,
        test_envs=test_envs,
        device=device,
        writer=writer,
        logger=logging
    )
    
    # Training loop
    start_time = time.time()
    
    for update in tqdm(range(1, config.num_updates + 1), desc="Training Updates"):
        # Train step
        train_metrics = trainer.train_step(update)
        
        # Log SPS
        sps = int(trainer.global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, trainer.global_step)
        
        # Save checkpoint
        if update % 30 == 0 or update == config.num_updates:
            save_checkpoint(agent, run_name, update)
        
        # Evaluate
        if update % 5 == 0 or update == config.num_updates:
            eval_metrics = evaluator.evaluate(trainer.global_step)
    
    # Cleanup
    train_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
