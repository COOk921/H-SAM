"""
Attention Model Wrapper for RL Agent.

This module provides the Agent class that wraps the attention-based
neural network for use with PPO training.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

from .nets.attention_model.attention_model import (
    AutoEmbedding,
    GraphAttentionEncoder,
    Decoder,
)
from .nets.attention_model.gat import GAT, ContainerHeteroGAT


class Problem:
    """Problem definition container."""
    
    def __init__(self, name: str):
        self.NAME = name


class Backbone(nn.Module):
    """Backbone encoder-decoder network."""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        problem_name: str = "tsp",
        n_encode_layers: int = 3,
        tanh_clipping: float = 10.0,
        n_heads: int = 8,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(
            self.problem.NAME, {"embedding_dim": embedding_dim}
        )

        self.embedding_dim = embedding_dim
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )

        self.decoder = Decoder(
            embedding_dim,
            self.embedding.context_dim,
            n_heads,
            self.problem,
            tanh_clipping
        )
        self.gat = ContainerHeteroGAT(7, self.embedding_dim, self.embedding_dim)

    def forward(self, obs):
        """Forward pass (not used in current implementation)."""
        return

    def encode(self, obs):
        """
        Encode observations.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            Cached embeddings for decoding
        """
        state = StateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input_data = state.states["observations"][:, :, 1:]

        """
        GAT encoding
        """
        b_graph = obs["graph_data"] 
        graph = Batch.from_data_list(b_graph)
        out = self.gat(graph.to(self.device))
        embedding = out.view(input_data.shape[0], input_data.shape[1], -1)
        encoded_inputs =  embedding


        # MHA encoding
        # embedding = self.embedding(input_data)
        # encoded_inputs, _ = self.encoder(embedding)
        
        cached_embeddings = self.decoder._precompute(encoded_inputs)
        return cached_embeddings

    def decode(self, obs, cached_embeddings):
        """
        Decode using cached embeddings.
        
        Args:
            obs: Observation dictionary
            cached_embeddings: Pre-computed embeddings from encode()
            
        Returns:
            Tuple of (logits, glimpse)
        """
        state = StateWrapper(obs, device=self.device, problem=self.problem.NAME)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)
        return logits, glimpse


class Actor(nn.Module):
    """Actor network that outputs action logits."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        logits = x[0]
        return logits


class Critic(nn.Module):
    """Critic network that estimates value."""
    
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out = self.mlp(x[1])
        return out


class Agent(nn.Module):
    """RL Agent combining backbone, actor, and critic."""
    
    def __init__(
        self, 
        embedding_dim: int = 128, 
        device: str = "cpu", 
        name: str = "tsp"
    ):
        super().__init__()
        self.backbone = Backbone(
            embedding_dim=embedding_dim, 
            device=device, 
            problem_name=name
        )
        self.critic = Critic(hidden_size=embedding_dim)
        self.actor = Actor()

    def forward(self, x):
        """
        Forward pass for inference.
        
        Args:
            x: Observation
            
        Returns:
            Tuple of (action, logits)
        """
        state = self.backbone.encode(x)
        x = self.backbone.decode(x, state)
        logits = self.actor(x)
        action = logits.max(2)[1]
        return action, logits

    def get_value(self, x):
        """Get value estimate."""
        x = self.backbone(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value."""
        x = self.backbone(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value_cached(self, x, state):
        """Get value using cached encoder state."""
        x = self.backbone.decode(x, state)
        return self.critic(x)

    def get_action_and_value_cached(self, x, action=None, state=None):
        """
        Get action and value using cached encoder state.
        
        Args:
            x: Observation
            action: Optional action (for computing log prob of given action)
            state: Cached encoder state
            
        Returns:
            Tuple of (action, log_prob, entropy, value, state)
        """
        if state is None:
            state = self.backbone.encode(x)
            x = self.backbone.decode(x, state)
        else:
            x = self.backbone.decode(x, state)

        logits = self.actor(x)
        logits = torch.clamp(logits, min=-1e9)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), state


class StateWrapper:
    """
    Wrapper to convert observation dictionary to state object.
    
    Converts numpy arrays to tensors and provides helper methods
    for the decoder.
    """

    def __init__(self, states: dict, device: str, problem: str = "tsp"):
        self.device = device
        
        self.states = {}
        for k, v in states.items():
            if k not in ('graph_data', 'data_key'):
                self.states[k] = torch.tensor(v, device=self.device)
            else:
                self.states[k] = v
        
        if problem == "container":
            self.is_initial_action = self.states["is_initial_action"].to(torch.bool)
            self.first_a = self.states["first_node_idx"]
        elif problem == "tsp":
            self.is_initial_action = self.states["is_initial_action"].to(torch.bool)
            self.first_a = self.states["first_node_idx"]
        elif problem == "cvrp":
            input_data = {
                "loc": self.states["observations"],
                "depot": self.states["depot"].squeeze(-1),
                "demand": self.states["demand"],
            }
            self.states["observations"] = input_data
            self.VEHICLE_CAPACITY = 0
            self.used_capacity = -self.states["current_load"]

    def get_current_node(self):
        """Get current node indices."""
        return self.states["last_node_idx"]

    def get_mask(self):
        """Get action mask."""
        return (1 - self.states["action_mask"]).to(torch.bool)
