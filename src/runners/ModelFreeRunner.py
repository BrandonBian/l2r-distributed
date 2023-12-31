"""Generalized runner for single-process RL. Takes in encoded observations, applies them to a buffer, and trains."""
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


@yamlize
class ModelFreeRunner(BaseRunner):
    """Main configurable runner."""

    def __init__(
        self,
        agent_config_path: str,
        buffer_config_path: str,
        env_config_path: str,
        model_save_dir: str,
        experiment_name: str,
        experiment_state_path: str,
        num_test_episodes: int,
        num_run_episodes: int,
        save_every_nth_episode: int,
        update_model_after: int,
        update_model_every: int,
        eval_every: int,
        max_episode_length: int,
        resume_training: bool = False,
    ):
        """Initialize ModelFreeRunner.

        Args:
            agent_config_path (str): Path to agent configuration YAML.
            buffer_config_path (str): Path to replay buffer configuration YAML.
            env_config_path (str): Path to the environment configuration YAML.
            model_save_dir (str): Path to save model
            experiment_name (str): Experiment name in WandB
            experiment_state_path (str): Path to save experiment state for resuming.
            num_test_episodes (int): Number of test episodes
            num_run_episodes (int): Number of training episodes
            save_every_nth_episode (int): Train save frequency ( episode ).
            update_model_after (int): Update model after some number of training steps.
            update_model_every (int): Update model every __ training steps, for ___ training steps.
            eval_every (int): Evaluate every ___ episodes.
            max_episode_length (int): Maximum episode length ( BAD PARAM / BUGGY. )
        """
        super().__init__()
        print("[Runner Init] ModelFreeRunner")
        # Moved initialzation of env to run to allow for yamlization of this class.
        # This would allow a common runner for all model-free approaches

        # Initialize runner parameters
        self.model_save_dir = model_save_dir
        self.num_test_episodes = num_test_episodes
        self.num_run_episodes = num_run_episodes
        self.save_every_nth_episode = save_every_nth_episode
        self.update_model_after = update_model_after
        self.update_model_every = update_model_every
        self.eval_every = eval_every
        self.max_episode_length = max_episode_length
        self.experiment_name = experiment_name
        self.experiment_state_path = experiment_state_path
        self.resume_training = resume_training

        if not self.experiment_state_path.endswith(".json"):
            raise ValueError(
                "Folder or incorrect file type specified. Expected json filename."
            )

        # AGENT Declaration
        self.agent = create_configurable(
            agent_config_path, NameToSourcePath.agent)

        # BUFFER Declaration
        if not self.resume_training:
            self.replay_buffer = create_configurable(
                buffer_config_path, NameToSourcePath.buffer
            )
            self.best_ret = 0
            self.last_saved_episode = 0
            self.best_eval_ret = 0

        else:
            with open(self.experiment_state_path, "r") as openfile:
                json_object = openfile.readline()
            running_vars = jsonpickle.decode(json_object)

            self.best_ret = running_vars["current_best_ret"]
            self.last_saved_episode = running_vars["last_saved_episode"]
            self.replay_buffer = running_vars["buffer"]
            self.best_eval_ret = running_vars["current_best_eval_ret"]

        self.env_wrapped = create_configurable(
            env_config_path, NameToSourcePath.environment)

    def run(self, api_key: str, exp_name: str):
        """Train an agent, with our given parameters, on the environment in question.

        Args:
            env (gym.env): Some gym-compliant environment, preferrably wrapped using a wrapper
            api_key (str, optional): Wandb API key for logging. Defaults to ''.
            exp_name (str, optional): Experiment name for logging in Wandb.
        """
        self.wandb_logger = None
        if api_key:
            self.wandb_logger = WanDBLogger(
                api_key=api_key, project_name="l2r", exp_name=exp_name
            )

        t = 0
        start_idx = self.last_saved_episode
        train_timer = time.time()

        for ep_number in range(start_idx + 1, self.num_run_episodes + 1):
            done = False

            obs_encoded = self.env_wrapped.reset(options={"random_pos": True})

            ep_ret = 0
            info = None

            while not done:
                t += 1
                self.agent.deterministic = False
                action = self.agent.select_action(obs_encoded)
                obs_encoded_new, reward, done, info = self.env_wrapped.step(action)

                ep_ret += reward

                self.replay_buffer.store(
                    {
                        "obs": obs_encoded,
                        "act": action,
                        "rew": reward,
                        "next_obs": obs_encoded_new,
                        "done": done,
                    }
                )
                if done or t == self.max_episode_length:
                    self.replay_buffer.finish_path(action)

                obs_encoded = obs_encoded_new
                if (t >= self.update_model_after) and (
                    t % self.update_model_every == 0
                ):
                    for _ in range(self.update_model_every):
                        batch = self.replay_buffer.sample_batch()
                        self.agent.update(data=batch)

            if ep_number % self.eval_every == 0:
                eval_ret = self.eval()
                if eval_ret > self.best_eval_ret:
                    self.best_eval_ret = eval_ret

            if self.wandb_logger:
                # TODO: revise try-except
                try:
                    # L2R
                    self.wandb_logger.log(
                        {
                            "reward": ep_ret,
                            "Distance": info["metrics"]["total_distance"],
                            "Time": info["metrics"]["total_time"],
                            "Num infractions": info["metrics"]["num_infractions"],
                            "Average Speed KPH": info["metrics"]["average_speed_kph"],
                            "Average Displacement Error": info["metrics"][
                                "average_displacement_error"
                            ],
                            "Trajectory Efficiency": info["metrics"][
                                "trajectory_efficiency"
                            ],
                            "Trajectory Admissability": info["metrics"][
                                "trajectory_admissibility"
                            ],
                            "Movement Smoothness": info["metrics"]["movement_smoothness"],
                            "Timestep per Sec": info["metrics"]["timestep/sec"],
                            "Laps Completed": info["metrics"]["laps_completed"],
                            "Success Rate": info['metrics']['success_rate']
                        }
                    )
                except:
                    # Non-L2R
                    train_duration = time.time() - train_timer
                    train_timer = time.time()

                    self.wandb_logger.log(
                        {
                            "Episode Reward": ep_ret,
                            "Episode Duration": train_duration
                        }
                    )

            # self.checkpoint_model(ep_ret, ep_number)

    def eval(self):
        """Evaluate model on the evaluation environment, using a deterministic agent if possible.

        Args:
            None

        Returns:
            float: The max reward for each test session.
        """
        val_ep_rets = []

        # Not implemented for logging multiple test episodes
        # assert self.cfg["num_test_episodes"] == 1

        for j in range(self.num_test_episodes):

            eval_obs_encoded = self.env_wrapped.reset(
                options={"random_pos": False})

            eval_done, eval_ep_ret, eval_ep_len, eval_n_val_steps, self.metadata = (
                False,
                0,
                0,
                0,
                {},
            )

            experience, t_eval = [], 0

            while (not eval_done) & (eval_ep_len <= self.max_episode_length):
                # Take deterministic actions at test time
                self.agent.deterministic = True
                self.t = 1e6
                eval_action = self.agent.select_action(eval_obs_encoded)
                (
                    eval_obs_encoded_new,
                    eval_reward,
                    eval_done,
                    eval_info,
                ) = self.env_wrapped.step(eval_action)

                # Check that the camera is turned on
                eval_ep_ret += eval_reward
                eval_ep_len += 1
                eval_n_val_steps += 1

                eval_obs_encoded = eval_obs_encoded_new
                t_eval += 1

            val_ep_rets.append(eval_ep_ret)

            # TODO: revise try-except
            if self.wandb_logger:
                try:
                    # L2R
                    self.wandb_logger.log(
                        {
                            "Eval reward": eval_ep_ret,
                            "Eval Distance": eval_info["metrics"]["total_distance"],
                            "Eval Time": eval_info["metrics"]["total_time"],
                            "Eval Num infractions": eval_info["metrics"]["num_infractions"],
                            "Evaluation Speed (KPH)": eval_info["metrics"][
                                "average_speed_kph"
                            ],
                            "Eval Average Displacement Error": eval_info["metrics"][
                                "average_displacement_error"
                            ],
                            "Eval Trajectory Efficiency": eval_info["metrics"][
                                "trajectory_efficiency"
                            ],
                            "Eval Trajectory Admissability": eval_info["metrics"][
                                "trajectory_admissibility"
                            ],
                            "Eval Movement Smoothness": eval_info["metrics"][
                                "movement_smoothness"
                            ],
                            "Eval Timesteps per second": eval_info["metrics"][
                                "timestep/sec"
                            ],
                            "Eval Laps completed": eval_info["metrics"]["laps_completed"],
                            "Eval Success Rate": eval_info['metrics']['success_rate']
                        }
                    )
                except:
                    # Non-L2R
                    self.wandb_logger.log(
                        {
                            "Eval Reward": eval_ep_ret,
                        }
                    )

        return max(val_ep_rets)

    def checkpoint_model(self, ep_ret, ep_number):
        """Conditionally save a checkpoint of the model if return is larger than best, or if self.save_every_nth_episode episodes have passed.

        Args:
            ep_ret (float): Current return
            ep_number (int): Current episode number
        """
        # Save every N episodes or when the current episode return is better than the best return
        # Following the logic of now deprecated checkpoint_model
        if ep_number % self.save_every_nth_episode == 0 or ep_ret > self.best_ret:
            self.best_ret = max(ep_ret, self.best_ret)
            save_path = f"{self.model_save_dir}/{self.experiment_name}/best_{self.experiment_name}_episode_{ep_number}.statedict"
            self.agent.save_model(save_path)
            # self.save_experiment_state(ep_number)

    def save_experiment_state(self, ep_number):
        """Save running variables for experiment state resuming.

        Args:
            ep_number (int): Current episode number

        Raises:
            Exception: Must specify json file name with experiment state path.
            Exception: Must specify experiment state path
        """
        running_variables = {
            "last_saved_episode": ep_number,
            "current_best_ret": self.best_ret,
            "buffer": self.replay_buffer,
            "current_best_eval_ret": self.best_eval_ret,
        }

        if self.experiment_state_path:
            # encoded = jsonpickle.encode(self)
            encoded = jsonpickle.encode(running_variables)
            with open(self.experiment_state_path, "w") as outfile:
                outfile.write(encoded)

        else:
            raise Exception("Path not specified or does not exist")

        pass
