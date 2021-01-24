import torch
import torch.nn as nn
from torch import optim
import numpy as np

device = torch.device("cuda:0")

class ModelRL(nn.Module):
    
    def __init__(self,learning_rate,h,n_states,n_actions,rank):

        super(ModelRL, self).__init__()

        self.linearK1 = nn.Linear(n_states+n_actions,rank,bias=False)
        self.linearD1 = nn.Linear(rank,h)
        self.linearK2 = nn.Linear(h, rank,bias=False)
        self.linearD2 = nn.Linear(rank,h)
        self.linearK3 = nn.Linear(h, rank, bias = False)
        self.linearD3 = nn.Linear(rank,n_states+1)

        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)

    def forward(self, x):

        x = torch.tanh(self.linearD1(self.linearK1(x)))
        x = torch.tanh(self.linearD2(self.linearK2(x)))
        x = self.linearD3(self.linearK3(x))

        return x

class ValueFunction(nn.Module):

    def __init__(self,width,n_states):
        
        super(ValueFunction, self).__init__()
        
        self.linear1 = nn.Linear(n_states,width)
        nn.init.normal_(self.linear1.weight,0.0,1/np.sqrt(n_states))
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        
        self.linear2 = nn.Linear(width,1)
        nn.init.normal_(self.linear2.weight,0.0,1/np.sqrt(width))
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        
    def forward(self, x):
        
        x = torch.tanh(self.linear1(x))
        value = self.linear2(x)
        
        return value


class DQN(nn.Module):
    
    def __init__(self,learning_rate,width,n_states,n_actions,epsilon):
        
        super(DQN, self).__init__()

        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_states = n_states
        
        self.linear1 = nn.Linear(n_states,width)
        nn.init.normal_(self.linear1.weight,0.0,1/np.sqrt(n_states))
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        
        self.linear2 = nn.Linear(width,n_actions)
        nn.init.normal_(self.linear2.weight,0.0,1/np.sqrt(width))
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)

    def forward(self, x):
        
        x = torch.tanh(self.linear1(x))
        q_value = self.linear2(x)
        
        return q_value
    
    def action(self,x):
        x = torch.tensor(x).to(device)
        r = np.random.random()
        if r<self.epsilon:
            action = np.random.randint(0,self.n_actions)
        else:
            with torch.no_grad():
                q_val = self.forward(x.float())
                action = torch.argmax(q_val).item()
        return action