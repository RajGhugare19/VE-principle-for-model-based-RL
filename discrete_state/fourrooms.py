import numpy as np
import random
from PIL import Image
import cv2
import time

class FourRooms():

    def __init__(self, size = 11):
        
        self.size = size
        self.n_actions = 4
        self.n_states = 68
        self.max_steps = 100
        self.n_steps = 0

        self.colors = {0: (0,0,0),           #blank(object id = 0)
                        1: (105, 105, 105),  #wall(object id = 1)
                        2: (0, 255, 0),      #goal(object id = 2)
                        3: (255, 0, 0)       #agent(object id = 3)
                        }          
        
        self.env = np.zeros([size,size])
        self.place_walls()
        self.place_goal()
        self.place_agent()
        self.features,self.index = self.get_features()


    def place_walls(self):

        self.env[0,:] = 1
        self.env[:,0] = 1
        self.env[-1,:] = 1
        self.env[:,-1] = 1
        self.env[5,:] = 1
        self.env[:,5] = 1
        self.env[(3,5,7,5),(5,3,5,7)] = 0
    
    def place_goal(self):
        
        self.gx,self.gy = 1,self.size-2
        self.env[1,self.size-2] = 2

    def place_agent(self):
        
        done = False
        #Rejection sampling
        while not done:
            x,y = np.random.randint(1,self.size-1,2)
            if self.env[x,y] == 0:
                self.env[x,y] = 3
                self.ax,self.ay = x,y
                done = True

    def step(self, action, prob=0.9):
        
        terminal = False

        p = np.random.uniform(0, 1)
        if p > 0.9:
            #choosing random action with prob 0.1
            action = self.sample_action()

        if action == 0:
            reward = self.move(x=0, y=1)          #EAST
        elif action == 1:
            reward = self.move(x=0,y=-1)          #WEST 
        elif action == 2:                
            reward = self.move(x=-1,y=0)          #NORTH
        elif action == 3:                
            reward = self.move(x=1, y=0)          #SOUTH

        if self.n_steps >= 100:
            terminal = True

        return self.index[self.ax,self.ay],reward,terminal

    def move(self,x,y):

        reward = 0
        next_x = self.ax + x
        next_y = self.ay + y

        if self.env[next_x,next_y] == 0:
            #next state is blank
            self.clear_block(self.ax,self.ay)
            self.place_block(3,next_x,next_y)
            self.ax = next_x
            self.ay = next_y

        elif self.env[next_x,next_y] == 2:    
            #next state is goal
            self.clear_block(self.ax,self.ay)
            self.place_agent()
            reward = 1

        elif self.env[next_x,next_y] == 1:
            #next state is wall
            pass

        else:
            print('Something wrong')

        self.n_steps += 1
        return reward

    def clear_block(self,x,y):
        
        self.env[x,y] = 0
    
    def place_block(self,obj,x,y):
        
        self.env[x,y] = obj

    def reset(self):
        
        self.n_steps = 0
        self.env = np.zeros([self.size,self.size])
        self.place_walls()
        self.place_goal()
        self.place_agent()
        return self.index[self.ax,self.ay]

    def render(self, render_time=100):

        img = np.zeros((self.size*31, self.size*31, 3), dtype=np.uint8)

        for i in range(self.size):
            for j in range(self.size):
                obj = self.env[i,j]
                img = self.fill(img,i,j,obj)

        img = Image.fromarray(img, 'RGB')
        cv2.imshow("image", np.array(img))
        cv2.waitKey(render_time)

    def fill(self, m, x, y, c, s=31):

        t = (s-1)//2
        x_t = (x)*s + t
        y_t = (y)*s + t
        m[x_t-t:x_t+t,y_t-t:y_t+t] = self.colors[c]
        return m

    def sample_action(self):

        return np.random.randint(0, self.n_actions)
    
    def get_features(self):

        features = np.zeros([68,2])
        index = np.zeros([self.size,self.size],dtype=int)
        ind = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.env[row,col] != 1:
                    features[ind] = [row,col]
                    index[row,col] = ind
                    ind += 1
        return features,index
    
    def state_features(self):

        return self.features