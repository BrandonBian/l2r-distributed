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

    def step(self, action):
        return self.env.step(action)

    def reset(self, action, options=None):
        return self.env.reset(action)

    def __getattr__(self, name):
        try:
            return getattr(self.env, name)
        except Exception as e:
            raise e
