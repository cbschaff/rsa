# Residual Policy Learning for Shared Autonomy

This repository accompanies the paper [Residual Policy Learning for Shared Autonomy](https://arxiv.org/abs/2004.05097). 

This repository contains code to train our shared autonomy agent with several human models, including agents trained to
imitate humans and corrupted versions of optimal policies. We have provided all of the human models used in our paper
as well as code for you to make your own human models. Trained shared autonomy agents are also provided.

You can interact with the models through a joystick and the pygame interface.

The code is based on my [deep reinforcement learning library](https://github.com/cbschaff/dl) which builds on PyTorch,
OpenAI Gym, gin-config, and Tensorboard.

## Installation

1. Install [docker](https://docs.docker.com/get-docker/).
2. Install [x-docker](https://github.com/afdaniele/x-docker), a wrapper around docker for running GUI applications inside a container.
3. In the top level directory, build the docker image by running:
    ```./build_docker.sh```
4. Launch the docker container by running:
    ```./launch_docker.sh```
    This will start a container and mount the code at ```/root/pkgs/residual_policy_learning```.
 
 All commands, unless otherwise stated, should be run from inside a docker container.
 
## Repository structure
The code is organized into the following directories:

* ```ppo/```: The code for training and interacting with shared autonomy agents.
* ```lunar_lander/```: The code for LunarLander, LunarReacher, and training RL agents for these environments.
* ```drone_sim/```: The code for DroneReacher and training RL agents for that environment.
* ```imitation_learning/```: The code for collecting human data and training behavioral cloning agents.
 
## Playing the games with and without our assistants

You can interact with the environments using a joystick controller (Xbox, etc.).

NOTE: The controls are hard coded for our particular joystick (Xbox one controller).
You may need to edit ```lunar_lander/joystick_agent.py``` and ```drone_sim/joystick_agent.py```
to match your joystick.

NOTE: Docker seems to only recognize a joystick if it is connected to your computer before you start the container.

### Lunar Lander and Lunar Reacher

In these environments, you control the thrusters of a spaceship. The controls are as follows:
* Tilting the RIGHT joystick VERTICALLY will control the main thruster.
* Tilting the RIGHT joystick HORIZONTALLY will control the side thrusters.

To play the game, run:

```
cd /root/pkgs/residual_shared_autonomy/lunar_lander
python joystick_agent.py
```

To play the game with our pretrained assistant, run:

```
cd /root/pkgs/residual_shared_autonomy/ppo
python joystick_control.py models/pretrained_lunar
```
To play LunarReacher, add a ```--reacher``` flag to the python commands above.

### Drone Reacher

In this environment, you will control a simulated drone to reacher a red floating target. You will control thrust and
angular acceleration. The controls are as follows:
* Tilt the LEFT joystick VERTICALLY to control thrust (the middle of the range outputs 1 g of thrust)
* Tilt the LEFT joystick HORIZONTALLY to control yaw
* Tilt the RIGHT joystick VERTICALLY to control pitch
* Tilt the RIGHT joystick HORIZONTALLY to control roll

To play the game, run:

```
cd /root/pkgs/residual_shared_autonomy/drone_sim
python joystick_agent.py
```

To play the game with our pretrained assistant, run:

```
cd /root/pkgs/residual_shared_autonomy/ppo
python joystick_control.py models/pretrained_drone --drone
```

## Training shared autonomy agents

Shared autonomy agents can be trained using the following commands:

```
cd /root/pkgs/residual_shared_autonomy/ppo
./train.sh logs configs/ppo_lunar.gin
```

Replace ```configs/ppo_lunar.gin``` with ```./configs/ppo_drone.gin``` to train the agent for the DroneReacher environment.

Experiments can be visualized by running:

```tensorboard --logdir /path/to/log/directory```

To change hyperparameters, edit the ".gin" files.

To play with your assistants, follow the instructions above.

To train assistants with laggy or noisy models, call ```train.sh``` with the 'laggy' and 'noisy' gin files.

## Creating human models

We provide the human models used in our paper, but you can create your own as well.

### Imitating human behavior

To create your own models, you will need to collect data and then train an agent to imitate that data.

To collect data, run:

```
cd /root/pkgs/residual_shared_autonomy/imitation_learning
python data_collection_lunar.py path/to/save/data num_episodes
```

Replace 'lunar' with 'drone' to collect data for the drone simulator.

To train a model with the collected data, first edit ```bc_lunar.gin``` by changing
BCTrainer.datafile to your data. Then run:

```
cd /root/pkgs/residual_shared_autonomy/imitation_learning
./train.sh /path/to/log/directory bc_lunar.gin
```

To view logs, run: ```tensorboard --logdir /path/to/logdir```

To observe your trained agent, run: ```python watch.py /path/to/logdir```. Add a ```--drone``` flag for the drone simulator.

To use your models in training a shared autonomy agent, place all of their log directories in a common directory
(or add them to the correct "behavioral_cloning_agents" directory in ```ppo/models```). Then change ```ppo/ppo_lunar.gin``` or 
```ppo/ppo_drone.gin``` so that ```BCMultiActor.logdir = "path/to/your/directory"```.

### Modelling humans as corrupted experts

We can also model humans as corrupted experts. In our paper, be explore laggy and noisy experts, where the expert is 
a policy trained using RL. We have included pretrained policies, but you can train your own using the following instructions.

For Lunar Lander, run:

```
cd /root/pkgs/residual_shared_autonomy/lunar_lander
./train.sh
```

For Drone Reacher, run:

```
cd /root/pkgs/residual_shared_autonomy/drone_sim
./train.sh
```

To train shared autonomy agents with these models, edit the '.gin' files in ```ppo/configs``` to point to your model
instead of the pretrained one by setting ```LunarLanderActor.logdir = "/path/to/your/log/directory"``` or 
```DroneReacherActor.logdir = "/path/to/your/log/directory"```.
