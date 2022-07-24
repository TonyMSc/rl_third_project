# Background
The objective of this project is to see if a double-jointed arm can move to target locations. A reward of +0.1 is provid6ed for each step that the agent's hand is in the goal location. We want the agent to maintain its position at the target location for as many time steps as possible.

We will be using the enviornment that has 20 identical arms


## Environment Details

The state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. The action space is a vector with four continuous numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Solved is defined as the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
Training for this model is very long and slow.

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

The following Hyperparameters are kept fixed for the first experiment that failed:
BUFFER_SIZE = int(1e6)  # replay buffer size \
BATCH_SIZE = 512 #128        # minibatch size \
GAMMA = 0.925            # discount factor \
TAU = 1e-3              # for soft update of target parameters \
LR_ACTOR = 1e-4         # learning rate of the actor \
LR_CRITIC = 1e-4        # learning rate of the critic, this was lowered from 3e-4 and learning improved \
WEIGHT_DECAY = 0.0000   # L2 weight decay, this was lowered from 0.0001 and lerning improved \
n_episodes=2000		 maximum number of episodes \
max_t=1000		 maximum number of timesteps per episode, this was increased from 700 

The following Hyperparameters are kept fixed for the first experiment that failed:
BUFFER_SIZE = int(1e6)  # replay buffer size \
BATCH_SIZE = 512 #128        # minibatch size \
GAMMA = 0.975            # discount factor \
TAU = 1e-3              # for soft update of target parameters \
LR_ACTOR = 1e-4         # learning rate of the actor \
LR_CRITIC = 1e-3        # learning rate of the critic\
WEIGHT_DECAY = 0.0000   # L2 weight decay, this was lowered from 0.0001 and lerning improved \
n_episodes=2000		 maximum number of episodes \
max_t=1000		 


The following Algorithm was tested. 

# Learning Algorithm
 
## DDPG
### Learning Algorithm
We use DDPG to solve controlled tasks with continuous action spaces.  The number of valid actions is infinite, so it would be extremely difficult to find the highest Q value.  To solve DDPG assume the $Q(s,a(s))$ is differentiable with respect to $a(s)$.  The policy will be represented by a neural network

We optimize with two neural networks (one for the policy and one for the Q values), we select an action in a specific state and use a 2nd neural network to compute the Q value of that state and action.  We compute the value of the action selected by the policy, and move the parameters of the policy in the direction of the maximum value increase (ie. The gradient).  Instead of an epsilon greedy, we add some gaussian noise to the policy. 

Steps
1.	Initialize the neural network that will estimate the Q values given a state and an action (the critic network) $Q(s,a|\theta^{Q})$ and the neural network that will take the action (the actor) $\mu(s|\theta^{\mu})$
2.	Initialize the target networks
3.	Create a replay buffer
4.	For a given episodes we take the observation of the environment and select and action from the policy and apply some random noise (instead of the greedy epsilon policy).  The action is taken in the environment and receive information about the reward and the next state, which is then stored in the replay buffer
5.	We select a mini batch of samples from the replay buffer and update the neural networks by computing the targets and computing the mse of the loss function and updating the weights by stochastic gradient descent 
6.	We update the policy by taking the action selected by the policy at each time step and computing the Q value of that state action pair using the Q Network to get an estimate of the performance
7.	Then apply stochastic gradient ascent to move the policy in the direction of higher growth
8.	The target networks are updated by using a certain percentage from the main network and a certain percentage from the target network.  We use Tau for that percentage split. 

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
