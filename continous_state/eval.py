import torch
import numpy as np

from models import DQN
from cartpole import CartPole
from arguments import get_args


env = CartPole()
n_states = env.n_states
n_actions = env.n_actions

epsilon = 0.05
DQN_lr = 5e-4

args = get_args(mode='eval')
value_width = args.value_width
rank_model = args.rank_model
exp = args.exp

if args.gpu != 'cpu':
    device = 'cuda:0'  #Running on first GPU
else:
    device = 'cpu'

model_path = 'pretrained/dqn/' + 'dqn' + exp + str(rank_model) + '_' + str(value_width) + '.pt'
Q_model = DQN(DQN_lr,value_width,n_states,n_actions,epsilon).to(device)
Q_model.load_state_dict(torch.load(model_path))



scores = []

for i in range(100):
    s = env.reset()
    done = False
    score = 0
    while not done:
        s_ = torch.tensor(s).to(device)
        q_val = Q_model(s_.float())
        a = torch.argmax(q_val).item()
        ns,r,done,_ = env.step(a)
        score += r
        s = ns
    print('episode: ',i)
    print('score: ', score)
    scores.append(score)
print("Average score for 100 runs is == ", np.mean(scores))
