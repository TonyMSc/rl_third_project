# Project Details
This is the third project for the Deep Reinforcement Learning Nanodegree.  

# Objective
The objective of this project is to have two agents bounce a ball back and forth over a net.  If an agent hits the ball over the net, it receives a reward of +0.1.  If the ball hits the ground or goes out of bounds, it receives a negative reward of -0.01.  We want to keep the ball in play for as long as possible.

## Environment Details
* State Space - consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 
* Action Space - Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

<br> When the agent must achieve an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents) the objective is met.

# Getting Started
Step1:
Install Anaconda distribution at:
https://www.anaconda.com/

Step2:
Follow the instructions to setup the environment for your system (window, mac, etc.) \
https://github.com/udacity/deep-reinforcement-learning.git

Step3:
After the virtual enviornment has been installed follow the instuctions for setting up the enviorment
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet

Step4:
Clone my project repo

```bash
git clone https://github.com/TonyMSc/rl_third_project.git
```

Step4:
Copy the Tennis.ipynb notebook, model.py, maddpg_agent.py files cloned from the repo and move them to /deep-reinforcement-learning/p3_collab-compet/ folder from the environment you created in Step 2 instructions.


# Instructions
Open the Navagation_main.ipynb notebook and change the file "path env = UnityEnvironment(file_name=".../p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe")". You should be able to run all the cells (graphs will print out).  


