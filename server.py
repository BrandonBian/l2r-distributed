from src.config.yamlize import NameToSourcePath, create_configurable
import threading
import sys
import os

# Training Paradigm - Distributed Collection (DistribCollect)
from distrib_l2r.asynchron.distribCollect.learner import DistribCollect_AsyncLearningNode
# Training Paradigm - Distributed Update (DistribUpdate)
from distrib_l2r.asynchron.distribUpdate.learner import DistribUpdate_AsyncLearningNode

if __name__ == "__main__":

    # Define the RL agent
    agent_name = os.getenv("AGENT_NAME").strip()
    print(f"Learner Initialized: '{agent_name}'")

    # Define the training paradigm
    training_paradigm = os.getenv("TRAINING_PARADIGM").strip()
    print(f"Training Paradigm Configured - '{training_paradigm}'")

    if agent_name == "walker":
        # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/

        if training_paradigm == "distribCollect":
            learner = DistribCollect_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
        elif training_paradigm == "distribUpdate":
            learner = DistribUpdate_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
        else:
            raise NotImplementedError
    elif agent_name == "mcar":
        # https://mgoulao.github.io/gym-docs/environments/classic_control/mountain_car_continuous/

        if training_paradigm == "distribCollect":
            learner = DistribCollect_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
        elif training_paradigm == "distribUpdate":
            learner = DistribUpdate_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    learner.learn()
