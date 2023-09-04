from src.config.yamlize import NameToSourcePath, create_configurable
import threading
import sys
import os

# Training Paradigm - Distributed Collection (DistribCollect)
from distrib_l2r.asynchron.distribCollect.learner import DistribCollect_AsyncLearningNode
# Training Paradigm - Distributed Update (DistribUpdate)

if __name__ == "__main__":

    # Define the RL agent
    agent_name = os.getenv("AGENT_NAME").strip()
    print(f"Learner Initialized: '{agent_name}'")

    # Define the training paradigm
    training_paradigm = os.getenv("TRAINING_PARADIGM").strip()
    print(f"Training Paradigm Configured - '{training_paradigm}'")

    if agent_name == "walker":
        # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/

        learner = DistribCollect_AsyncLearningNode(
            agent=create_configurable(
                "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
            ),
            api_key=sys.argv[1],
        )
    elif agent_name == "mcar":
        # https://mgoulao.github.io/gym-docs/environments/classic_control/mountain_car_continuous/

        learner = DistribCollect_AsyncLearningNode(
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
