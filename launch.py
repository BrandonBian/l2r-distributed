import subprocess
import yaml

############################
# - Input Configurations - #
############################
# Define the RL environment ('mcar', 'walker', 'l2r')
RL_env = input("Select RL environment (mcar/walker/l2r): ").strip()

# Define the training paradigm ('sequential', 'dCollect', 'dUpdate')
training_paradigm = input("Select training paradigm (sequential/dCollect/dUpdate): ").strip()

# Define number of workers (distributed only)
num_workers = "NA"
if training_paradigm != "sequential":
    num_workers = int(input("Input number of distributed workers: ").strip())

# Define experiment name to be logged on wandb
exp_name = input("Input WandB experiment name: ").strip()

# Sanity check
assert RL_env in ("mcar", "walker", "l2r")
assert training_paradigm in ("sequential", "dCollect", "dUpdate")

# Fetch the corresponding source file
if training_paradigm == "sequential":
    source_file = "./kubernetes/template-sequential.yaml"
else:
    source_file = "./kubernetes/template-distributed.yaml"

#############################
# - Output Configurations - #
#############################
print("----------")
print(f"RL Environment = [{RL_env}] | Training Paradigm = [{training_paradigm}] | Number of Workers = [{num_workers}] | Experiment Name = [{exp_name}] ---")
print("----------")

with open(source_file, "r") as file:
    data = yaml.safe_load(file)

######################################
# - Configure Template: Sequential - #
######################################
if training_paradigm == "sequential":
    # Configure metadata names (i.e., fill in the TODOs in the template YAML)
    data["metadata"]["name"] = f"{RL_env}-sequential"
    data["metadata"]["labels"]["tier"] = f"{RL_env}-sequential"
    data["spec"]["containers"][0]["name"] = f"{RL_env}-sequential"

    # Configure command
    command = data["spec"]["containers"][0]["command"][2]
    
    # NOTE: for L2R, we need to add commands to auto-launch Arrival simulator
    if RL_env == "l2r":
        command += "cd /home/LinuxNoEditor/ && "
        command += "sudo -u ubuntu ./ArrivalSim.sh -OpenGL & "
        command += "sleep 15m && "
        command += "cd / && "
        command += "cd l2r-distributed && "
        command += "git checkout sequential && "
    
    command += f"python3.8 -m scripts.main --env {RL_env} "
    command += "--wandb_apikey 173e38ab5f2f2d96c260f57c989b4d068b64fb8a "
    command += f"--exp_name {exp_name}"
    
    data["spec"]["containers"][0]["command"][2] = command


# Save the newly created kubernetes file
with open(f"./kubernetes/{RL_env}-{training_paradigm}.yaml", "w") as file:
    yaml.dump(data, file)
 
# Launch Kubernetes source file
subprocess.run(["kubectl", "create", "-f", f"./kubernetes/{RL_env}-{training_paradigm}.yaml"], check=True)
