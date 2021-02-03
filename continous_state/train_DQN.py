import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models import DQN, ModelRL, ValueFunction
from cartpole import CartPole
from utils import Memory
from arguments import get_args

env = CartPole()
n_states = env.n_states
n_actions = env.n_actions

args = get_args(mode='dqn')
value_width = args.value_width
rank_model = args.rank_model
exp = args.exp

if args.gpu != 'cpu':
    device = 'cuda:0'  #Running on first GPU
else:
    device = 'cpu'

#HPs are as given in the paper Value Equivalence Principle for Model-Based Reinforcement Learning
num_learn_steps = 2500000
num_learn_freq = 4
target_update = 2500
epsilon = 0.05
dqn_gamma = 0.99
gamma = 0.99
h_model = 256
batch_size = 32
DQN_lr = 5e-4
mem_size = 1000000
learning_rate = 5e-5
rank_model = rank_model
value_width = value_width


#initialising Q and Target Q models
Q_model = DQN(DQN_lr,value_width,n_states,n_actions,epsilon).to(device)
Q_model_target = DQN(DQN_lr,value_width,n_states,n_actions,epsilon).to(device)
Q_model_target.load_state_dict(Q_model.state_dict())
Q_model_target.eval()

replay_buffer = Memory(env,mem_size,device)
model = ModelRL(learning_rate,h_model,n_states,n_actions,rank_model).to(device)

if exp == 'VE':    
    path =  'pretrained/VE/' + exp + str(rank_model) + "_" + str(value_width) + '.pt'

elif exp == 'MLE':
    path = 'pretrained/MLE/' + exp + str(rank_model) + "_" + str(value_width) + '.pt'

model.load_state_dict(torch.load(path))
model.eval()

log = 'runs/dqn_' + exp + str(rank_model) + '_' + str(value_width) + '/'
writer = SummaryWriter(log)

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'saved')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

batch_index = np.arange(batch_size)
done = False
s = env.reset()
score = 0
best_avg = 0
scores = []

for n in range(1,num_learn_steps+1):

    a = Q_model.action(s)
    ns,r,done,_ = env.step(a)
    score += r
    one_hot_a = torch.tensor(np.eye(n_actions)[a]).to(device)
    x = torch.cat((torch.tensor(s).to(device),one_hot_a)).to(device) 
    out = model(x.float()).detach().cpu().numpy()
    ns_hat = out[0:5]
    r_hat = out[-1]

    replay_buffer.store(s,a,r_hat,ns_hat,done)
    if done:
      scores.append(score)
      score = 0  
      s = env.reset()
      done = False
    else:
      s = ns
    
    if replay_buffer.memory_count < mem_size:
        m = replay_buffer.memory_count
    else:
        m = mem_size

    if replay_buffer.memory_count > batch_size and n%num_learn_freq==0 :

        batch = np.random.choice(m, batch_size, replace=False)
        s_batch = torch.tensor(replay_buffer.state_memory[batch],dtype=torch.float32).to(device)
        ns_batch = torch.tensor(replay_buffer.next_state_memory[batch],dtype=torch.float32).to(device)
        r_batch = torch.tensor(replay_buffer.reward_memory[batch],dtype=torch.float32).to(device)
        a_batch = torch.tensor(replay_buffer.action_memory[batch])
        term_batch = torch.tensor(replay_buffer.terminal_memory[batch],dtype=torch.int8).to(device)
        
        q_val = Q_model.forward(s_batch)[batch_index,a_batch]
        max_a = torch.argmax(Q_model.forward(ns_batch),dim=1).detach()
        next_q_val = Q_model_target(ns_batch)[batch_index,max_a].detach()

        q_target = r_batch + dqn_gamma*next_q_val*(1-term_batch) 
        loss = Q_model.criterion(q_val,q_target).to(device)

        if n%5000 == 0:
          print("i = ", n)
          print("best avg till now= ",best_avg )
                   
        Q_model.optimizer.zero_grad()
        loss.backward()
        Q_model.optimizer.step()
        Q_model.optimizer.zero_grad()

    if n%target_update == 0:
        Q_model_target.load_state_dict(Q_model.state_dict())

    if len(scores) == 100:
        avg = np.mean(scores)
        writer.add_scalar('avg_rewards', avg, n)
        if avg > best_avg:
            #Saving best model
            best_avg = avg
            dqn_path = 'saved/dqn' + exp + str(rank_model) + "_" + str(value_width) + '.pt'
            torch.save(Q_model.state_dict(),dqn_path)
        scores.pop(0)

