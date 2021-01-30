import numpy as np
from fourrooms import FourRooms
import torch

env = FourRooms()
print(env.state_features())

t = torch.tensor([1,2,3])
path = 'saved/sample.pt'
torch.save(t, path)

k = torch.load(path)
print(k)
print(k.dtype)