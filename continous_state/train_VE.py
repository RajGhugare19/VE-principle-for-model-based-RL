import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models import ModelRL, ValueFunction
from cartpole import CartPole
from utils import Memory
from arguments import get_args

exp = 'VE'

env = CartPole()
n_states = env.n_states
n_actions = env.n_actions

args = get_args()
value_width = args.value_width
rank_model = args.rank_model

if args.gpu != 'cpu':
    device = 'cuda:0'  #Running on first GPU
else:
    device = 'cpu'

#HPs are as given in the paper Value Equivalence Principle for Model-Based Reinforcement Learning
total_iters=1000000
gamma = 0.99
h_model = 256
batch_size = 32
learning_rate = 5e-5
mem_size = 1000000
exp = 'MLE'
        

mem = Memory(env,mem_size,device) 
mem.store_random()
mem.to_tensor()


VE_model = ModelRL(learning_rate,h_model,n_states,n_actions,rank_model).to(device)

writer = SummaryWriter('runs/'+ exp + str(rank_model) + '_' + str(value_width))

running_loss = 0

for i in range(total_iters):
    
    batch = np.random.choice(mem_size, batch_size, replace=False)
    s = mem.state_memory[batch]
    ns = mem.next_state_memory[batch]
    r = mem.reward_memory[batch]
    a = mem.action_memory[batch]
    one_hot_a = mem.one_hot_action[batch]
    
    
    x = torch.cat((s,one_hot_a),dim=1)
    
    k_ve = VE_model(x)

    state_pred_batch = k_ve[:,0:5]
    reward_pred_batch = k_ve[:,-1]

    loss_ve = 0
    for j in range(5):
        V = ValueFunction(value_width,n_states).to(device)
        loss_ve += torch.norm(V(state_pred_batch) - V(ns))**2
    loss_ve += torch.norm(reward_pred_batch-r)**2
    
    running_loss += loss_ve.item()

    VE_model.optimizer.zero_grad()
    loss_ve.backward()
    VE_model.optimizer.step()
    VE_model.optimizer.zero_grad()

    if i%500 == 0:
        print("i = ", i)
        writer.add_scalar('training loss', running_loss/500, i)
        print("loss VE = ", running_loss/500)
        running_loss = 0

VE_path =  'saved/VE' + str(rank_model) + "_" + str(value_width) + '.pt'
torch.save(VE_model.state_dict(),VE_path)


