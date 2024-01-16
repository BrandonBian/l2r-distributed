# Learn-to-Race: Distributed RL with Optimizations

## Progress
- Every configuration is using **SAC agent** by default (Non-SAC configuration is not tested)
- For environments marked with "**OpenAI**", they are using implementation of SAC agents, buffers, and runners from [vanilla OpenAI SpinningUp repository](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac). For environments not marked, they are using implementation from prior iterations of this project, and may contain latent issues or deprecations. 
- **NOTE**: a table of all **OpenAI Gym Environments** can be found [HERE](https://github.com/openai/gym/wiki/Table-of-environments). Our implementation only supports environments with `BOX(x,)` for both observation and action spaces.
  - Mountain Car: `MountainCarContinuous-v0` (Observation Space = `Box([-1.2  -0.07], [0.6  0.07], (2,), float32)`, Action Space = `Box(-1.0, 1.0, (1,), float32)`)
  - Bipedal Walker: `BipedalWalker-v2` (Observation Space = `Box(-inf, inf, (24,), float32)`, Action Space = `Box(-1.0, 1.0, (4,), float32)`)
  - Lunar Lander: `LunarLanderContinuous-v2` (Observation Space = `Box(-inf, inf, (8,), float32)`, Action Space = `Box(-1.0, 1.0, (2,), float32)`)
  - L2R (learn-to-race): Our own environment (Observation Space = `Box(-inf, inf, (33,), float32)`, Action Space = `Box(-1.0, 1.0, (2,), float32)`)
- **NOTE**: &#10004; means convergence, &#10003; means non-convergence. If no checkmark, not currently implemented.


| RL Environments         | Sequential                   | Distributed Collection | Distributed Update |
| ----------------------- | ---------------------------- | ---------------------- | ------------------ |
| Mountain Car            | &#10004;                     | &#10004;               | &#10004;           |
| Bipedal Walker          | &#10003;                     | &#10003;               | &#10003;           |
| Bipedal Walker - OpenAI | &#10003;                     | &#10003;               | &#10003;           |
| Lunar Lander - OpenAI   | &#10004;                     | &#10004;               | &#10003;           |
| Learn-to-race           | &#10004; (rough convergence) | &#10003;               | &#10003;           |


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

# Delete pod (force delete, NOT RECOMMENDED since resource may be persisting!)
kubectl delete pod <pod-name> --force --grace-period=0
```

## Visualization using WandB
```bash
# > Register an account on W&B (https://wandb.ai/)
# > Get your API key (mine is: 173e38ab5f2f2d96c260f57c989b4d068b64fb8a)
# > Replace the KEY in the "<file-name>.yaml" files, like this
python3 server.py 173e38ab5f2f2d96c260f57c989b4d068b64fb8a

# > Replace the project_name (currently 'l2r') in this line from "distrib_l2r/asynchron/learner.py", with your own project name (created on W&B)
self.wandb_logger = WanDBLogger(api_key=api_key, project_name="l2r")
```

## Running sequential (non-distributed) L2R with ArrivalSim - manually within Phoebe Kubernetes pods
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