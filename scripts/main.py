from l2r import build_env

# from l2r import RacingEnv
from src.config.yamlize import NameToSourcePath, create_configurable
import sys
import logging
import os
from gym import Wrapper
import gym


if __name__ == "__main__":

    agent_name = os.getenv("AGENT").strip()
    print(f"Using RL Agent - '{agent_name}'")

    # Build environment
    if agent_name == "l2r":
        

        runner = create_configurable(
            "config_files/l2r_sac/runner.yaml", NameToSourcePath.runner
        )

    elif agent_name == "mcar":

        runner = create_configurable(
            "config_files/mcar_sac/runner.yaml", NameToSourcePath.runner
        )

    elif agent_name == "walker":
        env = gym.make("BipedalWalker-v3")

        runner = create_configurable(
            "config_files/walker_sac/runner.yaml", NameToSourcePath.runner
        )
    
    else:
        print(">> Incorrect agent name!")
        exit(1)

    with open(
        f"{runner.model_save_dir}/{runner.experiment_name}/git_config",
        "w+",
    ) as f:
        f.write(" ".join(sys.argv[1:3]))
    # Race!
    try:
        import torch

        torch.autograd.set_detect_anomaly(True)
        runner.run(env, "173e38ab5f2f2d96c260f57c989b4d068b64fb8a")
    except IndexError as e:
        logging.warning(e)
        runner.run(env, "")
