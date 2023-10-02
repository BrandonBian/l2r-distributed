import subprocess

# Define the RL environment ('mcar', 'walker', 'l2r')
RL_env = input("Select RL environment (mcar/walker/l2r): ").strip()

# Define the training paradigm ('sequential', 'dCollect', 'dUpdate')
training_paradigm = input("Select training paradigm (sequential/dCollect/dUpdate): ").strip()

# Sanity check
assert RL_env in ("mcar", "walker", "l2r")
assert training_paradigm in ("sequential", "dCollect", "dUpdate")

# Fetch the corresponding source file
source_file = RL_env + '-' + training_paradigm + '.yaml'

# Print the info
print("---")
print(f"--- Launching [{RL_env}] with training paradigm [{training_paradigm}]: kubernetes/{source_file} ---")
print("---")

# Launch Kubernetes source file
subprocess.run(["kubectl", "create", "-f", f"kubernetes/{source_file}"], check=True)
