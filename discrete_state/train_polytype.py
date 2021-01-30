import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch


from catch import Catch
from fourrooms import FourRooms
from utils import Memory,random_determininstic_policies,row_stochastic,policy_eval,rank_constrained, value_iteration
from arguments import get_args

env = Catch()

args = get_args('Polytype')
rank = args.rank_model
exp = args.exp
render = args.render
num_policies = args.num_policies
environment = args.env

if environment != 'F':
    env = Catch()
else:
    env = FourRooms()

if args.gpu != 'cpu':
    device = 'cuda:0'  #Running on first GPU
else:
    device = 'cpu'



#HPs are as given in the paper Value Equivalence Principle for Model-Based Reinforcement Learning
learning_rate = 5e-5
total_iters = 1000000
gamma = 0.99
mem_size = 1000000

mem = Memory(env,mem_size)
mem.store_random()

#Estimating reward and transition matrix
print("Collecting data...")
Reward = torch.zeros(env.n_states,env.n_actions)
Transition = torch.zeros(env.n_actions,env.n_states,env.n_states)
for i in range(mem_size):
    s,a,r,n_s = mem.state_memory[i],mem.action_memory[i],mem.reward_memory[i],mem.next_state_memory[i]
    Reward[s,a] += r
    Transition[a,s,n_s] += 1

z = torch.unsqueeze(torch.sum(Transition,axis=2),dim=2)
P_A = torch.divide(Transition,z) #unbiased transition matrix
R_A = torch.divide(Reward,torch.sum(Transition,axis=2).T) #unbiased reward matrix

writer = SummaryWriter('runs/polytype/'+ exp + str(rank) + '_' + str(num_policies))
running_loss = 0

if args.exp != 'MLE':

    
    F_D = (torch.rand(size=[env.n_actions,env.n_states,rank],device=device)*2 - 1).requires_grad_(True)
    F_K = (torch.rand(size=[env.n_actions,rank,env.n_states],device=device)*2 - 1).requires_grad_(True)
    policies = random_determininstic_policies(num_policies)
    v_p = policy_eval(policies,env,P_A,R_A,gamma)

    P_A = P_A.to(device)
    v_pi = torch.tensor(v_p,device=device).T
    T1 = torch.matmul(P_A,v_pi.float())

    for i in range(total_iters):
        
        P = rank_constrained(F_D,F_K)
        loss = torch.sum(torch.linalg.norm(T1-torch.matmul(P,v_pi.float()),dim=2)**2)
        loss.backward()
        with torch.no_grad():
            F_D -= learning_rate * F_D.grad
            F_K -= learning_rate * F_K.grad
        F_D.grad = None
        F_K.grad = None
        running_loss += loss.item()
        if i%25000 == 0:
            print("i = ", i)
            writer.add_scalar('training loss', running_loss/25000, i)
            print("VE loss = ", running_loss/25000)
            running_loss = 0

else:

    Transition = Transition.to(device)
    F_D = (torch.rand(size=[env.n_actions,env.n_states,rank],device=device)*2 - 1).requires_grad_(True)
    F_K = (torch.rand(size=[env.n_actions,rank,env.n_states],device=device)*2 - 1).requires_grad_(True)

    for i in range(total_iters):
        
        P = rank_constrained(F_D,F_K)
        loss = torch.sum(torch.multiply(Transition,-torch.log(P)))
        loss.backward()
        with torch.no_grad():
            F_D -= learning_rate * F_D.grad
            F_K -= learning_rate * F_K.grad
        F_D.grad = None
        F_K.grad = None
        running_loss += loss.item()
        if i%25000 == 0:
            print("i = ", i)
            writer.add_scalar('training loss', running_loss/25000, i)
            print("VE loss = ", running_loss/25000)
            running_loss = 0

v = value_iteration(env,P.detach().cpu().numpy(),R_A)
pi = np.zeros(env.n_states)
t = np.array(P.to('cpu').detach())
r = np.array(R_A.to('cpu').detach())
for s in range(env.n_states):
    v_temp = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        v_temp[a] = r[s,a] + gamma*np.dot(t[a,s,:],v) 
        pi[s] = np.argmax(v_temp)

v_fin = policy_eval([pi],env,P_A.cpu().numpy(),R_A,gamma)
print("The average state value for the output policy is ",np.mean(v_fin))

if render:
    for i in range(5):
        s = env.reset()
        done = False
        while not done:
            ns,r,done = env.step(pi[s])
            env.render()
            s = ns
            
