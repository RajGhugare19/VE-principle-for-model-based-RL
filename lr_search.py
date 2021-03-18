import torch
import numpy as np
import matplotlib.pyplot as plt

losses = {}
for j in [1,5]: 
    for i in range(2,6):
        loss_str = str(j) + 'e-' + str(i) + '.pt'
        losses[loss_str] = torch.load(loss_str).numpy()

n = np.arange(200)*(1000000/200)


plt.figure()
for j in [1,5]:
    for i in range(2,6):
        key = str(j) + 'e-' + str(i) + '.pt'
        plt.plot(n, losses[key], label = key)

plt.xlabel('number of iterations')
plt.ylabel('VE loss')
plt.title('learning_rate search')
plt.legend()
plt.show()