import numpy as np
import random
from PIL import Image
import cv2
import time


class Catch():

    def __init__(self, rows = 10, cols = 5): 

        self.max_steps = 100
        self.n_steps = 0
        self.n_actions = 3
        self.n_states = 250
        self.rows = rows
        self.cols = cols
        self.colors = {0: (0,0,0),           #blank(object id = 0)
                        1: (0, 255, 0),      #ball(object id = 2)
                        2: (255, 0, 0)       #agent(object id = 3)
                        } 
        self.env = np.zeros([self.rows,self.cols])
        self.place_goal()
        self.place_agent()

    def place_goal(self):
        
        self.gx = 0
        self.gy = np.random.randint(0,5)
        self.env[self.gx,self.gy] = 1
    
    def place_agent(self):
        
        self.ax = self.rows-1
        self.ay = np.random.randint(0,5)
        self.env[self.ax,self.ay] = 2
    
    def step(self, action):

        terminal = False

        if action == 0:
            reward = self.move(y=1)          #EAST
        elif action == 1:
            reward = self.move(y=-1)         #WEST 
        elif action == 2:                
            reward = self.move(y=0)          #NOTHING
        
        if self.n_steps >= 100:
            terminal = True

        return (self.ay*50+self.gx*5+self.gy),reward,terminal

    def move(self,y):

        reward = 0
        terminal = False
        next_y = self.ay + y

        if next_y == -1 or next_y == self.cols:
            pass
        else:
            self.clear_block(self.ax,self.ay)
            self.ay = next_y
            self.place_block(2,self.ax,self.ay) 
        
        if(self.gx==self.rows-1):
            #If the ball is at the bottom most row
            self.clear_block(self.gx,self.gy)
            self.place_goal()
        else:
            self.clear_block(self.gx,self.gy)
            self.gx += 1
            self.place_block(1,self.gx,self.gy) 

            if(self.gx==self.rows-1):
                if(self.gy==self.ay):
                    reward = 1
            

        self.n_steps += 1
        return reward

    def clear_block(self,x,y):
        
        self.env[x,y] = 0

    def place_block(self,obj,x,y):
        
        self.env[x,y] = obj

    def reset(self):
        
        self.n_steps = 0
        self.env = np.zeros([self.rows,self.cols])
        self.place_goal()
        self.place_agent()
        return (self.ay*50+self.gx*5+self.gy)
    
    def render(self, render_time=100):
        #cv2.destroyAllWindows()
        img = np.zeros((self.rows*31, self.cols*31, 3), dtype=np.uint8)

        for i in range(self.rows):
            for j in range(self.cols):
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
    
    def state_features(self):
        state = np.zeros([250,4])
        #Returns the coordinates of all states
        for i in range(250):
            state[i,0] = 9
            state[i,1] = i//50
            state[i,2] = ((i//5)%10)
            state[i,3] = i%5
        return state