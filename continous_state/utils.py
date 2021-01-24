import numpy as np
import torch

class Memory():

    def __init__(self, env, size, device):
        
        self.size = size
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.device = device
        self.state_memory = np.zeros([size,self.n_states],dtype = np.float32)
        self.next_state_memory = np.zeros([size,self.n_states],dtype = np.float32)
        self.action_memory = np.zeros(size,dtype=np.int)
        self.one_hot_action = np.zeros([size,self.n_actions],dtype=np.int)
        self.reward_memory = np.zeros(size,dtype=np.float32)
        self.terminal_memory = np.zeros(size,dtype=np.int8)
        self.memory_count = 0
    
    def store(self,state,action,reward,next_state,terminal):
        
        index = self.memory_count%self.size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.one_hot_action[index] = np.eye(self.n_actions)[action]
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
            action = np.random.randint(0,self.n_actions)
            next_state,reward,done,_ = self.env.step(action)
            self.store(state,action,reward,next_state,done)
            if done:
                state = self.env.reset()
                done = False
            else:
                state = next_state



    
