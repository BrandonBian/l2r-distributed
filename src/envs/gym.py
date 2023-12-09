"""Container for various OpenAI Gym Environments."""
import numpy as np
import torch
import itertools
from src.constants import DEVICE
from src.config.yamlize import create_configurable, yamlize, NameToSourcePath
import gym


@yamlize
class GymEnv:
    """Container for initializing Gym envs."""

    def __init__(self, env_name: str):
        """Initialize env 

        Args:
            env_name (str): Name of environement
        """
        self.env = gym.make(env_name)
        
        print("[Gym Init] Environment name:", env_name)
        print("[Gym Init] Environment observation space:", self.env.observation_space.shape)
        print("[Gym Init] Environment action space:", self.env.action_space.shape)

    def step(self, action):
        # NOTE: it seems like OpenAI Gym environment step() must take a numpy action
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        (obs_encoded_new, reward, done, info) = self.env.step(action)
        return torch.as_tensor(obs_encoded_new, device=DEVICE), reward, done, info

    def reset(self, options=None):
        return torch.as_tensor(self.env.reset(), device=DEVICE)

    def __getattr__(self, name):
        try:
            return getattr(self.env, name)
        except Exception as e:
            raise e
