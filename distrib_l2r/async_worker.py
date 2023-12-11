import numpy as np
from src.utils.envwrapper import EnvContainer
from src.constants import DEVICE
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import os
import time

from gym import Wrapper
import gym

from tianshou.data import ReplayBuffer
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import ParameterMsg
from distrib_l2r.utils import send_data
from src.constants import Task
from l2r import build_env

logging.getLogger('').setLevel(logging.INFO)

class AsnycWorker:
    """An asynchronous worker"""
    def __init__(
            self,
            env,
            runner,
            learner_address: Tuple[str, int],
            paradigm: Optional[str] = None,
    ) -> None:
        print("[AsyncLearningNode Init] Paradigm =", paradigm)
        self.learner_address = learner_address
        self.mean_reward = 0.0
        self.paradigm = paradigm
        self.env = env
        self.runner = runner

        print("[AsyncLearningNode Init] Environment Action Space ==", self.env.action_space)

    def work(self) -> None:
        counter = 0
        is_train = True

        print("Sending init message to establish connection")
        response = send_data(data=InitMsg(), addr=self.learner_address, reply=True)
        policy_id, policy = response.data["policy_id"], response.data["policy"]
        print("Finish init message, start true communication")

        if self.paradigm == "dUpdate":
            task = response.data["task"]
            print(f"dUpdate Worker: [{task}] | Param. Ver. = {policy_id}")
        
        if self.paradigm == "dUpdate":
            while True:
                """ Process request, collect data """
                if task == Task.TRAIN:
                    parameters = self.train(policy_weights=policy, batches=response.data["replay_buffer"])
                else:
                    buffer, result = self.process(policy_weights=policy, task=task)

                """ Send response back to learner """
                if task == Task.COLLECT:
                    """ Collect data, send back replay buffer (BufferMsg) """
                    response = send_data(
                        data=BufferMsg(data=buffer),
                        addr=self.learner_address,
                        reply=True
                    )

                    print(f"dUpdate Worker: [Task.COLLECT] | Param. Ver. = {policy_id} | Collected Replay Buffer = {len(buffer)}")

                elif task == Task.EVAL:
                    """ Evaluate parameters, send back reward (EvalResultsMsg) """
                    response = send_data(
                        data=EvalResultsMsg(data=result),
                        addr=self.learner_address,
                        reply=True,
                    )

                    reward = result["reward"]
                    print(f"dUpdate Worker: [Task.EVAL] | Param. Ver. = {policy_id} | Eval Reward = {reward}")

                else:
                    """ Train parameters on the obtained replay buffers, send back updated parameters (ParameterMsg) """
                    response = send_data(data=ParameterMsg(data=parameters), addr=self.learner_address, reply=True)
                    
                    duration = parameters["duration"]
                    print(f"dUpdate Worker: [Task.TRAIN] | Param. Ver. = {policy_id} | Training time = {duration} s")

                policy_id, policy, task = response.data["policy_id"], response.data["policy"], response.data["task"]

        elif self.paradigm == "dCollect":
            while True:
                buffer, result = self.collect_data(policy_weights=policy, is_train=is_train)
                self.mean_reward = self.mean_reward * (0.2) + result["reward"] * 0.8

                try:
                    if is_train:
                        response = send_data(
                            data=BufferMsg(data=buffer),
                            addr=self.learner_address,
                            reply=True
                        )

                        print(f"----- dCollect Worker Iteration {counter}: Training -----")
                        print(f">> Train Reward (not sent): {self.mean_reward}")
                        print(f">> Buffer Size (sent): {len(buffer)}")

                    else:
                        response = send_data(
                            data=EvalResultsMsg(data=result),
                            addr=self.learner_address,
                            reply=True,
                        )

                        print(f"----- dCollect Worker Iteration {counter}: Evaluation -----")
                        print(f">> Eval Reward (sent): {self.mean_reward}")
                        print(f">> Buffer Size (not sent): {len(buffer)}")

                except:
                    print("[Warning] Worker failed to send data to learner, waiting 5 seconds and re-trying!")
                    time.sleep(5)
                    continue

                is_train = response.data["is_train"]
                policy_id, policy = response.data["policy_id"], response.data["policy"]
                
                counter += 1
        else:
            raise NotImplementedError

    def collect_data(
            self, policy_weights: dict, is_train: bool = True
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""

        buffer, result = self.runner.run(self.env, policy_weights, is_train=is_train)

        return buffer, result
    
    def process(
            self, policy_weights: dict, task: Task
    ) -> Tuple[ReplayBuffer, Any]:
        """ Collect 1 episode of data (replay buffer OR reward) in the environment """
        buffer, result = self.runner.run(self.env, policy_weights, task=task)
        return buffer, result

    def train(self, policy_weights: dict, batches: list):
        """ Perform update of the received parameters based on all the received batches """
        parameters = self.runner.train(policy_weights, batches)
        return parameters