import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class actor_mod(nn.Module):
  def __init__(self,n_actions, input_dims, fc1_dims, fc2_dims):
    super(actor_mod,self).__init__()
    self.linear1 = nn.Linear(1,2).double()
    self.linear2 = nn.Linear(2,2).double()
    self.linear3 = nn.Linear(2,1).double()
  
  def forward(self,x):
    x=F.relu(self.linear1(x))
    x=F.relu(self.linear2(x))
    x=self.linear3(x)
    return x