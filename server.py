from src.config.yamlize import NameToSourcePath, create_configurable
import threading
import sys
import argparse

# Training Paradigm - Distributed Collection (DistribCollect)
from distrib_l2r.asynchron.distribCollect.learner import DistribCollect_AsyncLearningNode
# Training Paradigm - Distributed Update (DistribUpdate)
from distrib_l2r.asynchron.distribUpdate.learner import DistribUpdate_AsyncLearningNode

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

    parser.add_argument(
        "--wandb_apikey",
        type=str,
        help="Enter your Weights-And-Bias API Key."
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        help="Enter your experiment name, to be recorded by Weights-And-Bias."
    )

    args = parser.parse_args()
    print(f"Server Configured - '{args.env}'")
    print(f"Training Paradigm Configured - '{args.paradigm}'")

    if args.env == "walker":
        # https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
        if args.paradigm == "dCollect":
            learner = DistribCollect_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        elif args.paradigm == "dUpdate":
            learner = DistribUpdate_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        else:
            raise NotImplementedError
        
    elif args.env == "mcar":
        # https://mgoulao.github.io/gym-docs/environments/classic_control/mountain_car_continuous/
        if args.paradigm == "dCollect":
            learner = DistribCollect_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        elif args.paradigm == "dUpdate":
            learner = DistribUpdate_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        else:
            raise NotImplementedError
    
    elif args.env == "l2r":
        if args.paradigm == "dCollect":
            learner = DistribCollect_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_l2r/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        elif args.paradigm == "dUpdate":
            learner = DistribUpdate_AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_l2r/agent.yaml", NameToSourcePath.agent
                ),
                api_key=args.wandb_apikey,
                exp_name=args.exp_name
            )
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError

    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    learner.learn()
