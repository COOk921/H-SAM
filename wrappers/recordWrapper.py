"""
Episode Statistics Recording Wrapper.

This module provides a wrapper for recording episode statistics
including returns and lengths.
"""

import time
from collections import deque
from typing import Tuple, Dict, Any

import gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    """Wrapper that records episode statistics."""
    
    def __init__(self, env: gym.Env, deque_size: int = 100):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            deque_size: Size of the statistics queue
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.n_traj = env.n_traj
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs) -> Any:
        """Reset the environment and statistics."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros((self.num_envs, self.n_traj), dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.finished = [False] * self.num_envs
        return observations

    def step(self, action) -> Tuple[Any, np.ndarray, Any, Dict]:
        """
        Take a step and record statistics.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations, rewards, dones, infos = super().step(action)

        self.episode_returns += rewards
        self.episode_lengths += 1
        
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)
        
        for i in range(len(dones)):
            if dones[i].all() and not self.finished[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return.copy(),
                    "l": episode_length,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                self.finished[i] = True

        if self.is_vector_env:
            infos = tuple(infos)
        
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )
