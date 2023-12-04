from src.runners.base import BaseRunner
import json
import time
import numpy as np
import wandb
from src.loggers.WanDBLogger import WanDBLogger
from src.runners.base import BaseRunner
from src.utils.envwrapper import EnvContainer

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE

from torch.optim import Adam
import gym
import torch
import itertools
import jsonpickle

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
        # Run control
        steps_per_epoch: int, 
        epochs: int, 
        start_steps: int, 
        max_ep_len: int,
        update_after: int,
        update_every: int,
        batch_size: int,
        num_test_episodes: int
    ):
        
        # Fetch agent, replay buffer, and environment wrapper
        self.agent = create_configurable(agent_config_path, NameToSourcePath.agent)
        self.replay_buffer = create_configurable(buffer_config_path, NameToSourcePath.buffer)
        self.env_wrapped = create_configurable(env_config_path, NameToSourcePath.environment)
        self.test_env_wrapped = create_configurable(env_config_path, NameToSourcePath.environment)

        # Save parameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.max_ep_len = max_ep_len
        self.update_after = update_after
        self.update_every = update_every
        self.batch_size = batch_size
        self.num_test_episodes = num_test_episodes

        # Initialize agent
        self.agent.init_network(
            obs_space=self.env_wrapped.env.observation_space,
            action_space=self.env_wrapped.env.action_space,
        )

    def run(self, api_key: str, exp_name: str):
        # Set wandb
        self.wandb_logger = WanDBLogger(
            api_key=api_key, project_name="l2r", exp_name=exp_name
        )

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env_wrapped.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.agent.select_action(o)
            else:
                a = self.env_wrapped.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env_wrapped.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                print("Episode reward:", ep_ret)
                o, ep_ret, ep_len = self.env_wrapped.reset(), 0, 0
                
            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.agent.update(data=batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                eval_ret = eval()

                print(f"[Epoch = {epoch}] Eval Reward: {eval_ret}")


    def eval(self):
        for _ in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env_wrapped.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env_wrapped.step(self.agent.select_action(o, True))
                ep_ret += r
                ep_len += 1
        
        return ep_ret


            



# batch_size=100,
# start_steps=10000, 
# update_after=1000,
# update_every=50,
# num_test_episodes=10,
# max_ep_len=1000, 