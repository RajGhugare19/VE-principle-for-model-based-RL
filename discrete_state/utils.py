import numpy as np
import torch
from sklearn.cluster import KMeans


def row_stochastic(rows,cols):
    m = torch.nn.Softmax(dim=2)
    P = torch.matmul(m(F_D),m(F_K))
    return P

def rank_constrained(F_D,F_K):
    m = torch.nn.Softmax(dim=2)
    P = torch.matmul(m(F_D),m(F_K))
    return P

class Memory():
    def __init__(self,env,size,device):
        
        self.size = size
        self.env = env
        self.state_memory = np.zeros([size],dtype = np.int)
        self.next_state_memory = np.zeros([size],dtype = np.int)
        self.action_memory = np.zeros(size,dtype=np.int)
        self.one_hot_action = np.zeros([size,env.n_actions],dtype=np.int)
        self.reward_memory = np.zeros(size,dtype=np.float32)
        self.terminal_memory = np.zeros(size,dtype=np.int8)
        self.memory_count = 0
    
    def store(self,state,action,reward,next_state,terminal):
        
        index = self.memory_count%self.size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.one_hot_action[index] = np.eye(3)[action]
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = terminal
        self.memory_count+=1 
    
    def to_tensor(self):
        self.state_memory = torch.tensor(self.state_memory).to(self.device)
        self.next_state_memory = torch.tensor(self.next_state_memory).to(self.device)
        self.one_hot_action = torch.tensor(self.one_hot_action).to(self.device)
        self.reward_memory = torch.tensor(self.reward_memory).to(self.device)

    def store_random(self):
        state = self.env.reset()
        done = False
        for i in range(self.size):
            action = np.random.randint(0,self.env.n_actions)
            next_state,reward,done = self.env.step(action)
            self.store(state,action,reward,next_state,done)
            if done:
                state = self.env.reset()
                done = False
            else:
                state = next_state

def random_determininstic_policies(num=20,n_states=250,n_actions=3):
    policies = []
    for i in range(num):
        pol = np.random.randint(0,n_actions,(n_states))
        policies.append(pol)
    return policies

def policy_eval(policies,env,t,r,gamma=0.99):
    v_pi = []
    for policy in policies:
        v = np.zeros(env.n_states)
        v_new = np.zeros(env.n_states)
        while True:
            for s in range(env.n_states):
                a = int(policy[s])
                v_new[s] = r[s,a] + gamma*np.dot(t[a,s,:],v)
            if np.max(np.abs(v - v_new)) < 0.001:
                break
            v = np.copy(v_new)
        v_pi.append(v)
    return v_pi

def value_iteration(env,t,r,gamma=0.99,epsilon=0.01):
    v = np.zeros(env.n_states)
    v_new = np.zeros(env.n_states)
    while True:
        for s in range(env.n_states):
            v_temp = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                v_temp[a] = r[s,a] + gamma*np.dot(t[a,s,:],v) 
            v_new[s] = np.max(v_temp)
        if np.max(np.abs(v - v_new)) < epsilon:
            break
        v = np.copy(v_new)
    return v

def get_savename(env,method,exp,rank,meta):
    pass

def get_kmeans(d,data):
    km = KMeans(
        n_clusters=d, init='k-means++',
        n_init=10,tol=1e-04)
    km.fit(data)
    return km

def collect_traj(size,policy,P_es,R_es):
    state = np.zeros(size,dtype=np.int)
    next_state = np.zeros(size,dtype=np.int)
    reward = np.zeros(size,dtype=np.int)
    action = np.zeros(size,dtype=np.int)
    term = np.zeros(size,dtype=np.int)
    s = env.reset()
    
    #Storing on-policy data
    for j in range(size):
        
        a = policy[s]
        n_s,r,t = env.step(a)
        n_s_hat = np.random.choice(state_list, 1, p=P_es[a,s])
        #n_s_hat = np.argmax(P_es[a,s])
        r_hat = R_es[s,a]
        state[j] = s
        action[j] = a
        next_state[j] = n_s_hat
        reward[j] = r_hat
        term[j] = t
        
        if t:
            s = env.reset()
        else:
            s = n_s
    return state,next_state,reward,action,term

def value_iteration(env,t,r,gamma=0.99,epsilon=0.01):
    v = np.zeros(env.n_states)
    v_new = np.zeros(env.n_states)
    t = np.array(t.to('cpu').detach())
    r = np.array(r.to('cpu').detach())
    while True:
        for s in range(env.n_states):
            v_temp = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                v_temp[a] = r[s,a] + gamma*np.dot(t[a,s,:],v) 
            v_new[s] = np.max(v_temp)
        if np.max(np.abs(v - v_new)) < epsilon:
            break
        v = np.copy(v_new)
    return v