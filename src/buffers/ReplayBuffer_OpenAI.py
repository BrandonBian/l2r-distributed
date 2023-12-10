import collections
import torch
import numpy as np
from typing import Tuple

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.agents.SAC_core import combined_shape

@yamlize
class ReplayBuffer_OpenAI:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        print("[Replay Buffer Init] ReplayBuffer - OpenAI")
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32, device=DEVICE)
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32, device=DEVICE)
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float32, device=DEVICE)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, values):
        original_size = self.size
        if type(values) is dict:
            self.obs_buf[self.ptr] = values['obs']
            self.obs2_buf[self.ptr] = values['next_obs']
            self.act_buf[self.ptr] = values['act']
            self.rew_buf[self.ptr] = values['rew']
            self.done_buf[self.ptr] = values['done']
            self.ptr = (self.ptr+1) % self.max_size
            self.size = min(self.size+1, self.max_size)
        elif type(values) == self.__class__:
            for idx in range(values.size):
                self.obs_buf[self.ptr] = values.obs_buf[idx]
                self.obs2_buf[self.ptr] = values.obs2_buf[idx]
                self.act_buf[self.ptr] = values.act_buf[idx]
                self.rew_buf[self.ptr] = values.rew_buf[idx]
                self.done_buf[self.ptr] = values.done_buf[idx]
                self.ptr = (self.ptr + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
        else:
            print(type(values), self.__class__)
            raise Exception(
                "Sorry, invalid input type. Please input dict or buffer of same type"
            )

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def __len__(self):
        # For compatbility with SimpleReplayBuffer
        return self.size
    
    def finish_path(self, action_obj=None):
        pass