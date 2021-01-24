import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models import ModelRL
from cartpole import CartPole
from utils import Memory
from arguments import get_args

exp = 'MLE'

env = CartPole()
n_states = env.n_states
n_actions = env.n_actions

args = get_args()
rank_model = args.rank_model
value_width = args.value_width

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


mem = Memory(env,mem_size,device) 
mem.store_random()
mem.to_tensor()

MLE_model = ModelRL(learning_rate,h_model,n_states,n_actions,rank_model).to(device)

writer = SummaryWriter('runs/'+ exp + str(rank_model) + '_' + str(value_width))

running_loss = 0
for i in range(1,total_iters):
    
    batch = np.random.choice(mem_size, batch_size, replace=False)
    s = mem.state_memory[batch]
    ns = mem.next_state_memory[batch]
    r = mem.reward_memory[batch]
    a = mem.action_memory[batch]
    one_hot_a = mem.one_hot_action[batch]
    
    
    x = torch.cat((s,one_hot_a),dim=1)
    
    k_mle = MLE_model(x)

    loss_mle = torch.sum(torch.square(k_mle-torch.cat((ns,r.reshape([-1,1])),dim=1)))
    MLE_model.optimizer.zero_grad()
    loss_mle.backward()
    MLE_model.optimizer.step()
    MLE_model.optimizer.zero_grad()    

    running_loss += loss_mle.item()

    if i%500 == 0:
        print("i = ", i)
        writer.add_scalar('training loss', running_loss/500, i)
        print("lose MLE = ", running_loss/500)
        running_loss = 0


MLE_path = 'saved/MLE' + str(rank_model) + "_" + str(value_width) + '.pt'
torch.save(MLE_model.state_dict(),MLE_path)

