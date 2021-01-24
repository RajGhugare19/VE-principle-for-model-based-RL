import sys 
import argparse 

def get_args(mode='train'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("value_width", help="number of nodes in the hidden layer of value function", type=int)
    parser.add_argument("rank_model", help="rank of the weight matrix of the model", type=int)
    parser.add_argument('-g', '--gpu', choices=[0, 'cpu'], help='GPU or CPU to be used for training', default=0)
    
    if mode == 'dqn':
        parser.add_argument("exp" , choices=['MLE', 'VE'], help="VE or MLE", type=str)

    if mode == 'eval':
        parser.add_argument("exp" , choices=['MLE', 'VE'], help="VE or MLE", type=str)

    args = parser.parse_args()

    return args
