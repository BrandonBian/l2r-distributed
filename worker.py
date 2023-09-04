import socket
import os

# Training Paradigm - Distributed Collection (DistribCollect)
from distrib_l2r.asynchron.distribCollect.worker import DistribCollect_AsnycWorker
# Training Paradigm - Distributed Update (DistribUpdate)

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
        worker = DistribCollect_AsnycWorker(learner_address=learner_address)
    elif agent_name == "mcar":
        learner_ip = socket.gethostbyname("mcar-learner-service")
        learner_address = (learner_ip, 4444)
        worker = DistribCollect_AsnycWorker(learner_address=learner_address)
    else:
        print("Invalid Agent!")
        exit(1)

    worker.work()
