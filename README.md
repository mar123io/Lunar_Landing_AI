# Lunar Lander Reinforcement Learning Agent

## Project Overview
This project involves training a reinforcement learning (RL) agent to solve the Lunar Lander environment from the OpenAI Gym.
The agent employs Deep Q-Learning (DQN) to learn how to land a spaceship on the lunar surface efficiently.
The environment provides visual feedback and the agent's performance is periodically recorded.

## Requirements
Ensure you have the following libraries installed before running the project:
- `numpy`
- `torch`
- `gymnasium`
- `imageio`

You can install these dependencies using:
- `pip install -r requirements.txt`

# Project Structure

## lunar_landing.py: 
- The main code file that contains the implementation of the RL agent and the training loop.

## video/:
- Directory where the recorded videos of the agent's performance are saved.

# Code Overview

## Network Class
- The Network class defines the neural network architecture used for the DQN agent. It consists of three fully connected layers with ReLU activations.

## ReplayMemory Class
- The ReplayMemory class implements a cyclic buffer to store experience tuples (state, action, reward, next_state, done). It samples random batches of experiences for training the network.

## Agent Class
- The Agent class encapsulates the behavior of the RL agent. Key components include:

## Networks:
- Local and target Q-networks.

## Optimizer: 
- Adam optimizer for updating the local network.

## Memory: 
- Replay memory to store experiences.

# Methods:

## step():
- Saves experience and triggers learning periodically.

## act():
- Chooses an action based on the current policy with epsilon-greedy exploration.

## learn(): 
- Updates the network using experiences from replay memory.

## soft_update():
- Softly updates the target network to gradually match the local network.

## Training the Agent
- The function train_agent trains the agent over a specified number of episodes. It initializes the environment, runs episodes, and adjusts epsilon for exploration-exploitation trade-off.

## Video Recording
- The RecordVideo wrapper records the agent's performance every 50 episodes. The videos are saved in the ./video directory.

## Displaying Video
- The function show_video checks if a video file exists in the specified path and notifies the user of its location.

# Running the Project

1. Clone the repository:

- `git clone https://github.com/yourusername/your-repository.git`

2. Navigate to the project directory.

3. Install the dependencies into your virtual environment:

- `pip install -r requirements.txt`
  
4. Run the main script to start training:

- `python lunar_landing.py`

# Results
- The training loop prints the average score every 100 episodes. The environment is considered solved when the agent achieves an average score of 300 over 100 consecutive episodes. The trained model's weights are saved as checkpoint.pth.
