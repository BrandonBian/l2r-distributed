import socket
from distrib_l2r.asynchron.worker import AsnycWorker
from src.config.yamlize import create_configurable

# from src.utils.envwrapper_aicrowd import EnvContainer
# from tianshou.policy import SACPolicy
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.continuous import ActorProb, Critic
import os

if __name__ == "__main__":

    # Define the RL agent
    agent_name = os.getenv("AGENT_NAME").strip()
    print(f"Worker Configured - '{agent_name}'")

    # Define the training paradigm
    training_paradigm = os.getenv("TRAINING_PARADIGM").strip()
    print(f"Training Paradigm Configured - '{training_paradigm}'")

    if agent_name == "walker":
        learner_ip = socket.gethostbyname("walker-learner-service")
        learner_address = (learner_ip, 4444)
        worker = AsnycWorker(learner_address=learner_address)
    elif agent_name == "mcar":
        learner_ip = socket.gethostbyname("mcar-learner-service")
        learner_address = (learner_ip, 4444)
        worker = AsnycWorker(learner_address=learner_address)
    else:
        print("Invalid Agent Name!")

    worker.work()
