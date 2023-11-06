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
from distrib_l2r.utils import send_data

from l2r import build_env

logging.getLogger('').setLevel(logging.INFO)

class DistribCollect_AsnycWorker:
    """An asynchronous worker"""
    def __init__(
            self,
            learner_address: Tuple[str, int],
            buffer_size: int = 5000,
            env_wrapper: Optional[Wrapper] = None,
            env_name: Optional[str] = None,
            **kwargs,
    ) -> None:
        self.learner_address = learner_address
        self.buffer_size = buffer_size
        self.mean_reward = 0.0

        # Build the environment and runner
        if env_name == "mcar":
            self.env = gym.make("MountainCarContinuous-v0")
            self.runner = create_configurable(
                "config_files/async_sac_mcar/worker.yaml", NameToSourcePath.runner
            )
        elif env_name == "walker":
            self.env = gym.make("BipedalWalker-v3")
            self.runner = create_configurable(
                "config_files/async_sac_walker/worker.yaml", NameToSourcePath.runner
            )
        elif env_name == "l2r":
            self.env = build_env(
                controller_kwargs={"quiet": True},
                env_kwargs=
                    {
                        "multimodal": True,
                        "eval_mode": True,
                        "n_eval_laps": 5,
                        "max_timesteps": 5000,
                        "obs_delay": 0.1,
                        "not_moving_timeout": 50000,
                        "reward_pol": "custom",
                        "provide_waypoints": False,
                        "active_sensors": [
                            "CameraFrontRGB"
                        ],
                        "vehicle_params":False,
                    },
                action_cfg=
                    {
                        "ip": "0.0.0.0",
                        "port": 7077,
                        "max_steer": 0.3,
                        "min_steer": -0.3,
                        "max_accel": 6.0,
                        "min_accel": -1,
                    },
                camera_cfg=[{
                        "name": "CameraFrontRGB",
                        "Addr": "tcp://0.0.0.0:8008",
                        "Width": 512,
                        "Height": 384,
                        "sim_addr": "tcp://0.0.0.0:8008",
                    }]
            )

            self.encoder = create_configurable(
                "config_files/async_sac_l2r/encoder.yaml", NameToSourcePath.encoder
            )
            self.encoder.to(DEVICE)

            self.env.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1.0, 1.0]))
            self.env = EnvContainer(self.encoder, self.env)

            self.runner = create_configurable(
                "config_files/async_sac_l2r/distribCollect_worker.yaml", NameToSourcePath.runner
            )

        else:
            raise NotImplementedError

        print("(worker.py) Action Space ==", self.env.action_space)

    def work(self) -> None:
        """Continously collect data"""
        counter = 0
        is_train = True
        logging.info("Sending init message to establish connection")
        response = send_data(
            data=InitMsg(), addr=self.learner_address, reply=True)
        policy_id, policy = response.data["policy_id"], response.data["policy"]
        logging.info("Finish init message, start true communication")

        while True:
            buffer, result = self.collect_data(
                policy_weights=policy, is_train=is_train)
            self.mean_reward = self.mean_reward * \
                (0.2) + result["reward"] * 0.8

            if is_train:
                response = send_data(
                    data=BufferMsg(data=buffer),
                    addr=self.learner_address,
                    reply=True
                )

                logging.info(f" --- Iteration {counter}: Training ---")
                logging.info(f" >> reward: {self.mean_reward}")
                logging.info(f" >> buffer size (sent): {len(buffer)}")

            else:

                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )

                logging.info(f" --- Iteration {counter}: Inference ---")
                logging.info(f" >> reward (sent): {self.mean_reward}")
                logging.info(f" >> buffer size: {len(buffer)}")

            is_train = response.data["is_train"]
            policy_id, policy = response.data["policy_id"], response.data["policy"]

            print("")
            counter += 1

    def collect_data(
            self, policy_weights: dict, is_train: bool = True
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""

        buffer, result = self.runner.run(self.env, policy_weights, is_train=is_train)

        return buffer, result
