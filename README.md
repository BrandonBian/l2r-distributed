# Learn-to-Race: Distributed RL with Optimizations

## Basic Kubernetes control commands
```bash
# Create pods
kubectl create -f <file-name>.yaml

# Check pod creation status
kubectl get pods
kubectl describe pod <pod-name>

# Check pod outputs (updates in real time)
kubectl logs -f <pod-name>

# Go inside a pod environment
kubectl exec -it <pod-name> -- /bin/bash

# Delete all pods
kubectl delete all --all

# Delete pod (force delete)
kubectl delete pod <pod-name> --force --grace-period=0
```

## Visualization using WandB
```bash
# > Register an account on W&B (https://wandb.ai/)
# > Get your API key (mine is: 173e38ab5f2f2d96c260f57c989b4d068b64fb8a)
# > Replace the KEY in the "<file-name>.yaml" files, like this
python3 server.py 173e38ab5f2f2d96c260f57c989b4d068b64fb8a

# > Replace the PROJECT NAME in this line from "distrib_l2r/asynchron/learner.py", with your own project name (created on W&B)
self.wandb_logger = WanDBLogger(api_key=api_key, project_name="test-project")
```

## Running sequential (non-distributed) L2R with ArrivalSim - within Phoebe Kubernetes pods
```bash
# Start kubernetes worker pod (fresh GPU environment)
kubectl create -f l2r-sequential.yaml

##############################################
# - Commands automatically ran upon launch - #
##############################################

# Clone repo
git clone https://github.com/BrandonBian/l2r
cd l2r/
git checkout sequential-l2r

# Install L2R framework
pip install git+https://github.com/learn-to-race/l2r@aicrowd-environment

# Install requirements
pip install -r setup/devtools_reqs.txt

# Resolve CV2 (OpenCV) circuar import issue
pip install "opencv-python-headless<4.3"

# Sleep indefinitely until user enters and manually executes

#################################################################
# - Wait until finish, then enter into pod to launch programs - #
#################################################################

# Enter into the worker-pod
kuebctl exec -it <worker-pod-name> -- /bin/bash

# Start ArrivalSim
cd LinuxNoEditor/
sudo -u ubuntu ./ArrivalSim.sh -OpenGL

# Run the script (on another terminal in the same worker pod)
cd l2r/
python3 -m scripts.main
```