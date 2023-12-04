"""This is OpenAI' Spinning Up PyTorch implementation of Soft-Actor-Critic with
minor adjustments.
For the official documentation, see below:
https://spinningup.openai.com/en/latest/algorithms/sac.html#documentation-pytorch-version
Source:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
"""
import itertools
from multiprocessing.sharedctypes import Value
from copy import deepcopy

import torch
import numpy as np
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize
from src.utils.utils import ActionSample
from src.constants import DEVICE
from src.agents.SAC_core import MLPActorCritic, count_vars

@yamlize
class SACAgent_OpenAI(BaseAgent):
    """Adopted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py"""

    def __init__(
        self,
        # Hyperparameters
        gamma: float, 
        polyak: float,
        alpha: float,
        lr: float,
        # Actor critic model
        seed=0,
        actor_critic=MLPActorCritic, 
        ac_kwargs=dict(), 
    ):
        super(SACAgent_OpenAI, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initializations
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr = lr
        self.actor_critic = actor_critic
        self.ac_kwargs = ac_kwargs
        
        self.determinisitic = False
        self.t = 0

    def init_network(self, obs_space, action_space):
        # Create actor-critic module and target networks
        self.actor_critic = self.actor_critic(obs_space, action_space, **self.ac_kwargs)
        self.actor_critic_target = deepcopy(self.actor_critic)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q1, self.actor_critic.q2])
        print('[SAC Init] Actor Critic Network - Number of parameters: pi: %d, q1: %d, q2: %d\n'%var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.lr)
        self.q_optimizer = Adam(q_params, lr=self.lr)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.actor_critic.q1(o,a)
        q2 = self.actor_critic.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critic_target.q1(o2, a2)
            q2_pi_targ = self.actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def select_action(self, obs):
        action_obj = ActionSample()

        a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
        if a.shape == ():
            # In case a is a scalar
            a = np.array([a])

        action_obj.action = a
        return action_obj

    def load_model(self, path_or_checkpoint):
        if isinstance(path_or_checkpoint, str):
            self.actor_critic.load_state_dict(torch.load(path_or_checkpoint))
        else:
            self.actor_critic.load_state_dict(path_or_checkpoint)

    def state_dict(self):
        return self.actor_critic.state_dict()

    def save_model(self, path):
        torch.save(self.actor_critic.state_dict(), path)
    
    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, _ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)