import socket
import os
import argparse

# Training Paradigm - Distributed Collection (DistribCollect)
from distrib_l2r.asynchron.distribCollect.worker import DistribCollect_AsnycWorker

# Training Paradigm - Distributed Update (DistribUpdate)
from distrib_l2r.asynchron.distribUpdate.worker import DistribUpdate_AsnycWorker

if __name__ == "__main__":

    # Argparse for environment + training paradigm selection and wandb config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["l2r", "mcar", "walker"],
        help="Select the environment ('l2r', 'mcar', or 'walker')."
    )

    parser.add_argument(
        "--paradigm",
        choices=["dCollect", "dUpdate"],
        help="Select the distributed training paradigm ('dCollect', 'dUpdate')."
    )
    

    args = parser.parse_args()
    print(f"Worker Configured - '{args.env}'")
    print(f"Training Paradigm Configured - '{args.paradigm}'")

    # Configure learner IP (by environment)
    if args.env == "mcar":
        learner_ip = socket.gethostbyname(f"mcar-{args.paradigm.lower()}")
        learner_address = (learner_ip, 4444)
    elif args.env == "walker":
        learner_ip = socket.gethostbyname(f"walker-{args.paradigm.lower()}")
        learner_address = (learner_ip, 4445)
    else:
        raise NotImplementedError
    
    # Configure worker (by training paradigm)
    if args.paradigm == "dCollect":
        worker = DistribCollect_AsnycWorker(learner_address=learner_address)
    elif args.paradigm == "dUpdate":
        worker = DistribUpdate_AsnycWorker(learner_address=learner_address)
    else:
        raise NotImplementedError

    worker.work()
