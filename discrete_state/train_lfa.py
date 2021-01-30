import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.cluster import KMeans

from catch import Catch
from fourrooms import FourRooms
from utils import Memory,row_stochastic,rank_constrained,get_kmeans
from arguments import get_args

env = Catch()

args = get_args('lfa')

rank = args.rank_model
exp = args.exp
render = args.render
d = args.num_features
environment = args.env

if environment != 'F':
    env = Catch()
else:
    env = FourRooms()

if args.gpu != 'cpu':
    device = 'cuda:0'  #Running on first GPU
else:
    device = 'cpu'


learning_rate = 5e-3
#HPs are as given in the paper Value Equivalence Principle for Model-Based Reinforcement Learning
total_iters = 1000000
gamma = 0.99
mem_size = 1000000

env = Catch()
features = env.state_features()
mem = Memory(env,mem_size)
mem.store_random()
data = features[mem.state_memory]

km = get_kmeans(d,data) 
ph_index = km.predict(features)
ph_feature = np.eye(d)[ph_index]


#Estimating reward and transition matrix
Reward = torch.zeros(env.n_states,env.n_actions)
Transition = torch.zeros(env.n_actions,env.n_states,env.n_states)
for i in range(1000000):
    s,a,r,n_s = mem.state_memory[i],mem.action_memory[i],mem.reward_memory[i],mem.next_state_memory[i]
    Reward[s,a] += r
    Transition[a,s,n_s] += 1
z = torch.unsqueeze(torch.sum(Transition,axis=2),dim=2)
P_A = torch.divide(Transition,z) #unbiased transition matrix
R_A = torch.divide(Reward,torch.sum(Transition,axis=2).T) #unbiased reward matrix

writer = SummaryWriter('runs/lfa/'+ exp + str(rank) + '_' + str(num_policies))
running_loss = 0

v_basis = torch.tensor(ph_feature, device = device)
P_A = P_A.to(device)
F_D = (torch.rand(size=[env.n_actions,env.n_states,rank],device=device)*2 - 1).requires_grad_(True)
F_K = (torch.rand(size=[env.n_actions,rank,env.n_states],device=device)*2 - 1).requires_grad_(True)

running_loss = 0

if args.exp != 'MLE':
    #VE
    T = torch.matmul(P_A,v_basis.float())
    for i in range(total_iters):
        P = rank_constrained(F_D,F_K)
        loss = torch.sum(torch.linalg.norm(T-torch.matmul(P,v_basis.float()),dim=2)**2)
        loss.backward()
        running_loss += loss.item()
        with torch.no_grad():
            F_D -= learning_rate * F_D.grad 
            F_K -= learning_rate * F_K.grad
        F_D.grad = None
        F_K.grad = None
        if i%25000 == 0:
            print("i = ", i)
            print("running loss = ", running_loss)
            writer.add_scalar('training loss', running_loss/25000, i)
            running_loss = 0
else:
    #MLE 
    for i in range(total_iters):
        P = rank_constrained(F_D,F_K)
        loss = torch.sum(torch.multiply(-P_A,torch.log(P)))
        loss.backward()
        running_loss += loss.item()
        with torch.no_grad():
            F_D -= learning_rate * F_D.grad 
            F_K -= learning_rate * F_K.grad 
        
        F_D.grad = None
        F_K.grad = None
        if i%25000 == 0:
            print("i = ", i)
            print("running loss = ", running_loss)
            writer.add_scalar('training loss', running_loss/25000, i)
            running_loss = 0