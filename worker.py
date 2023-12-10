import socket
import argparse

from src.config.yamlize import NameToSourcePath, create_configurable
from distrib_l2r.async_worker import AsnycWorker

if __name__ == "__main__":
    # Argparse for environment + training paradigm selection and wandb config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["l2r", "mcar", "walker", "walker-openai", "lander-openai"],
    )

    parser.add_argument(
        "--paradigm",
        choices=["dCollect", "dUpdate"],
        help="Select the distributed training paradigm ('dCollect', 'dUpdate')."
    )

    parser.add_argument(
        "--port",
        help="Enter the port name of the learner to connect with"
    )
    
    args = parser.parse_args()
    print(f"Worker Configured - '{args.env}'")
    print(f"Training Paradigm Configured - '{args.paradigm}'")
    print(f"Get host by name: {args.env}-{args.paradigm.lower()}-learner")

    # Configure learner IP (by environment)
    learner_ip = socket.gethostbyname(f"{args.env}-{args.paradigm.lower()}-learner")
    learner_address = (learner_ip, 4444)
    
    # Configure worker (by training paradigm)
    worker = AsnycWorker(
        # TODO: worker has its own copy of environment for collecting data
        env=create_configurable(f"config_files/{args.env}/env.yaml", NameToSourcePath.environment),
        runner=create_configurable(f"config_files/{args.env}/worker.yaml", NameToSourcePath.runner),
        learner_address=learner_address, 
        paradigm=args.paradigm
    )

    worker.work()
