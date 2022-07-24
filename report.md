# Background
The objective of this project is to have two agents bounce a ball back and forth over a net.  If an agent hits the ball over the net, it receives a reward of +0.1.  If the ball hits the ground or goes out of bounds, it receives a negative reward of -0.01.  We want to keep the ball in play for as long as possible.



## Environment Details

* State Space - consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 
* Action Space - Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

<br> When the agent must achieve an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents) the objective is met.

### Model Hyperparameters
An infinite number of hyperparameter combinations can be used in this problem.  Including:
1. number of hidden layers in the neural network
2. number of nodes in each layer in the neural network
3. the optimizer used in the neural network
4. the learning rate of the optimizer
5. starting epsilon for the epsilon greedy policy
6. decay rate of epsilon
7. the mini batch size
8. the buffer size for the replay
9. the gamma rate for the discount factor

Based on trial and error the following were used

The following Hyperparameters were used to train:
BUFFER_SIZE = int(1e6)  # replay buffer size \
BATCH_SIZE = 512        # minibatch size \
GAMMA = 0.99            # discount factor \
TAU = 5e-2              # for soft update of target parameters  \
LR_ACTOR = 1e-3         # learning rate of the actor \
LR_CRITIC = 1e-3        # learning rate of the critic, this was lowered from 3e-4 and learning improved \
WEIGHT_DECAY = 0.0000   # L2 weight decay, this was lowered from 0.0001 and lerning improved \
n_episodes=2000		 maximum number of episodes \

Changes in the learning rate and TAU caused the biggest change is how fast the problem was solved.  A higher TAU improved training.

The following Algorithm was tested. 

# Learning Algorithm
 
## MADDPG
### Learning Algorithm
We use Multi-Agent Deep Deterministic Policy Gradient (MADDPG) to solve controlled tasks with continuous action spaces.  The number of valid actions is infinite, so it would be extremely difficult to find the highest Q value.  

We are building on the DDPG algorithm

To solve DDPG assume the $Q(s,a(s))$ is differentiable with respect to $a(s)$.  The policy will be represented by a neural network

We optimize with two neural networks (one for the policy and one for the Q values), we select an action in a specific state and use a 2nd neural network to compute the Q value of that state and action.  We compute the value of the action selected by the policy, and move the parameters of the policy in the direction of the maximum value increase (ie. The gradient).  

The main change from DDPG to MADDPG is we now have cetralized training and decentralized execution.  The agents share experiencs in the experience replay buffer, but each agent only uses local observations at execution time.  During training actors choose actions, then information is then sent into the critic network to determine the Q values.  During execution the actors only uses local information, based on observation of the state space, and chooses the appropriate action based on policy that has been learned. 

### Model Architectures
The neural network architecture is a simple feed forward neural network (one for the Actor and one for the Critic):  
1. The inputs are the state size (for this problem it is a state space of 33)
2. The hidden layer consists of several fully connected linear layers with a relu activation function 
3. The output is the number of actions (or one value for the critic) we can take in the environment (for this problem the agent can take 4 actions in a continuous space from -1 to 1 for the Actor and 1 value from the Critic)
4. The optimizer for this network is Adam with a learning rate of 1e-4
5. The Critic loss function to minimize is the mean squared error of the $Q_{expected}$ and the $Q_{target}$
 

# Plot of Rewards from Experiments
Results from the experiments are as follows: \
# Analysis of results 
It was a surprise that the 20-arm model trained much faster than the one arm model.  However, if you look at it from a perspective of an ensemble method, then the more agents’ experiences to learn from the faster the neural net would learn and converge.  

This problem seemed very sensitive to changes in hyperparameters.  Lowering the learning rate and adding batch normalization improved performance.  When first training the model, it was difficult to get an average reward over 1.  After changing some hyperparameters and experimenting with different combinations of layers and nodes, training improved (but was very slow even with a gpu).

The first model the learning rate was set too low.  The increase was consistent, but training on a cpu took 4 days and after 140 episodes, the score started to decrease.  
![](images/proj2exp1.png)

The second attempt did much better when we changed the learning rate for the critic, reduced the number of nodes per layer in the neural net, and changed the noise to a normal distribution.
![](images/proj2exp2.png)
# Ideas for Future Work
**Neural Net Architecture**-Possibly use a CNN layer with the feed forward neural net to help identify colors.  Experiment with different number of layers and neuron combinations.  
**Aditional Experiments**-The following could also be used: 
1. PPO
2. A3C
3. D4PG
4. Twin Delayed DDPG
