from src.runners.base import BaseRunner

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE, Task

from typing import Optional
from torch.optim import Adam
from copy import deepcopy
import torch, time, gym


@yamlize
class OpenAIRunner():
    def __init__(
        self,
        # Configuration YAML files
        agent_config_path: str,
        buffer_config_path: str,
        env_config_path: str,
    ):
        
        # Fetch agent, replay buffer, and environment wrapper
        self.agent = create_configurable(agent_config_path, NameToSourcePath.agent)
        self.replay_buffer = create_configurable(buffer_config_path, NameToSourcePath.buffer)
        self.env_wrapped = create_configurable(env_config_path, NameToSourcePath.environment)

        # Initialize agent
        self.agent.init_network(
            obs_space=self.env_wrapped.env.observation_space,
            action_space=self.env_wrapped.env.action_space,
        )
        


# steps_per_epoch=4000, 
# epochs=100, 
# batch_size=100,
# start_steps=10000, 
# update_after=1000,
# update_every=50,
# num_test_episodes=10,
# max_ep_len=1000, 