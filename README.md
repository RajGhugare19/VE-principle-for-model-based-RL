# The Value Equivalence Principle for Model Based Reinforcement Learning

This repository is the official implementation of [The Value Equivalence Principle for Model based Reinforcement Learning](https://arxiv.org/abs/2011.03506)

You can find our report on the reproducibility [here](https://openreview.net/forum?id=IU5y7hIIZqS).

## Table of Contents
- [The Value Equivalence Principle for Model Based Reinforcement Learning](#the-value-equivalence-principle-for-model-based-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Continous State](#continous-state)
    - [Introduction](#introduction)
    - [Environments](#environments)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Pretrained Models](#pretrained-models)
  - [Discrete State](#discrete-state)
      - [Value Function Polytype](#value-function-polytype)
        - [Introduction](#introduction-1)
        - [Environment](#environments-1)
        - [Training and Evaluation](#training-and-evaluation)
      - [Linear Function Approximation](#linear-function-approximation)
        - [Introduction](#introduction-2)
        - [Environment](#environments-2)

## Requirements

- the dependencies are listed in the `requirements.yml` file
    ```
    conda env create -f environment.yml
    ```

## Continous State
### Introduction
This section contains the experiments for which we are enforcing VE principle with respect to a set of Neural Networks. Training code for VE and MLE is available. Evaluation of both methods is done by training a double DQN based policy.<br>
### Environments
A modified version of the classic CartPole environment is used. The code for which is available in our repository.<br>

### Training

Training arguments for train_MLE.py and train_VE.py:<br>
- value_width: number of nodes in the hidden layers of value function neural network.<br>
- rank_model: fixed rank for the weight matrices of the value function neural network.<br>

By default all models use cuda:0 as the device:<br>
- gpu(`-g`): 'cpu' for disabling cuda usage.

Additional arguments for train_DQN.py:<br>
- exp(`-e`): model to be used for training ddqn, i.e 'MLE' or 'VE'.

Train MLE model with rank 6 and width 128<br>
```
python3 train_MLE.py 128 6
```

Train VE model with rank 6 and width 128 using cpu<br>
```
python3 train_VE.py 128 6 -g 'cpu'
```
Train DDQN based policy using pretrained VE models of rank 4 and width 128<br>
```
python3 train_DQN.py 128 6 VE
```
### Evaluation
Many pretrained DDQN based polcies are available in the repository. One can check the performance using<br>
```
python3 eval.py 128 6 VE
```

### Pretrained Models
You can check the '/continous_state/pretrained' directory for all available pretrained pytorch models.


## Discrete State

## Value Function Polytype
### Introduction
This section contains the experiments for which we are enforcing VE principle with respect to a set true value functions. Training code for VE and MLE is available. Evaluation of both methods is done by forming a greedy policy using value-iteration.<br>
### Environments
Catch and FourRooms environment can be used for these sets of experiments. Altough we only reproduced the results on Catch.<br>

### Training and evaluation
Since these experiments take less than 10 minutes, we don't have a seperated evaluation module.
Required Arguments:
- rank_model: rank of the transition probability matrix.
- exp: name of method to be used.
- num_policies: Number of policies, whose value functions shall span of the set V.

Optional Arguments:
- e(`-e`): name of environment, i.e. 'Catch' or 'FourRooms'.
- r(`-r`): 1 for rendering the final policy on respective environment and 0 for not.
By default all models use cuda:0 as the device:<br>
- gpu(`-g`): 'cpu' for disabling cuda usage

train and evaluate a model with rank 30 and number of policies 40 using VE method
```
python3 train_polytype.py 30 VE 40
```

## Linear Function Approximation
### Introduction
This section contains the experiments for which we are enforcing the VE principle with respect to a set of Linear function approximators. Training code for VE and MLE is available. Evaluation of both methods can be done using two ways. First, using approximate policy iteration with LSTD. Second, using Double DQN with linear Q value functions.<br>
### Environments
Catch and FourRooms environment can be used for these sets of experiments. Altough we only reproduced the results on Catch.<br>

For simplicity the training and evaluation for these experiments was done using a jupyter-notebook. This includes clean code divided in different sections, for training VE, MLE models and evaluating them using approximate policy iteration using LSTD and using Double DQN.

