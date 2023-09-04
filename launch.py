import subprocess

# Define the RL agent ('mcar', 'walker', 'l2r')
RL_agent = "mcar"

# Define the training paradigm
training_paradigm = 'distCollect'

# Fetch the corresponding source file
source_file = RL_agent + '-' + training_paradigm + '.yaml'

# Print the info
print(f"--- Launching [{RL_agent}] with training paradigm [{training_paradigm}]: {source_file} ---")

# Launch Kubernetes source file
try:
    subprocess.run(["kubectl", "create", "-f", f"kubernetes/{source_file}"], check=True)
    print("Kubernetes resource applied successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error applying Kubernetes resource: {e}")