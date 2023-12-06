from src.config.yamlize import NameToSourcePath, create_configurable
import torch
import argparse

if __name__ == "__main__":

    # Argparse for environment selection and wandb config
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        choices=["l2r", "mcar", "walker", "walker-openai", "lander-openai"]
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

    # Initialize the runner and start run
    print(f"Environment: {args.env} | Experiment: {args.exp_name}")

    runner = create_configurable(f"config_files/{args.env}/runner.yaml", NameToSourcePath.runner)

    torch.autograd.set_detect_anomaly(True)
    runner.run(args.wandb_apikey, args.exp_name)
