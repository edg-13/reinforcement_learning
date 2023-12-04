import random
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
from gym.core import Env
from torch import nn

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        self.layers.apply(self.init_he)
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x
    
    def init_he(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()

def ddqn_loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    model_actions = policy_dqn(next_states).argmax(1).unsqueeze(-1)
    targets = (~dones).reshape(-1)*target_dqn(next_states).gather(1, model_actions).reshape(-1) + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    criterion = nn.HuberLoss(reduction='sum', delta=0.2)
    return criterion(q_values, targets.squeeze(-1))

def huber_loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Huber loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    criterion = nn.HuberLoss(reduction='sum', delta=0.2)
    return criterion(q_values, targets)

def vdbe_boltzmann(epsilon:float, loss:float, inv_sens:float, delta=0.5)->float:
    """Updates epsilon according to VDBE
    
    Args:
        epsilon: current epsilon
        loss: Bellman loss of the batch
        inv_sens: inverse sensitivity sigma
        delta: VDBE parameter
        
    Returns:
        float with updated epsilon
    """
        
    bfactor = np.exp(-loss/inv_sens)
    f = (1-bfactor)/(1+bfactor)
    
    return delta*f + (1-delta)*epsilon
    
    
    
    
    