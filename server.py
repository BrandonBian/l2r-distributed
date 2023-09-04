from distrib_l2r.asynchron.distCollect.learner import DistCollect_AsyncLearningNode
from src.config.yamlize import NameToSourcePath, create_configurable
# from tianshou.policy import SACPolicy
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.continuous import ActorProb, Critic
import threading
import sys
import os

if __name__ == "__main__":

    # Define the RL agent
    agent_name = os.getenv("AGENT_NAME").strip()
    print(f"Learner Initialized: '{agent_name}'")

    # Define the training paradigm
    training_paradigm = os.getenv("TRAINING_PARADIGM").strip()
    print(f"Training Paradigm Configured - '{training_paradigm}'")

    if agent_name == "walker":
        # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/

        learner = DistCollect_AsyncLearningNode(
            agent=create_configurable(
                "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
            ),
            api_key=sys.argv[1],
        )
    elif agent_name == "mcar":
        # https://mgoulao.github.io/gym-docs/environments/classic_control/mountain_car_continuous/

        learner = DistCollect_AsyncLearningNode(
            agent=create_configurable(
                "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
            ),
            api_key=sys.argv[1],
        )
    else:
        print("Invalid Agent!")
        exit(1)

    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    learner.learn()
