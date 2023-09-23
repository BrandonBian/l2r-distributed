from src.config.yamlize import NameToSourcePath, create_configurable
import torch

if __name__ == "__main__":
    runner = create_configurable(
        "config_files/l2r_sac/runner.yaml", NameToSourcePath.runner)
    
    torch.autograd.set_detect_anomaly(True)
    
    runner.run("173e38ab5f2f2d96c260f57c989b4d068b64fb8a")
