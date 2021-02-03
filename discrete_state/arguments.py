import sys 
import argparse 

def get_args(mode):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("rank_model", help="rank of the transition matrix", type=int)
    parser.add_argument('-g', '--gpu', choices=['0', 'cpu'], help='GPU or CPU to be used for training', default=0)
    parser.add_argument('-r','--render', choices=['0','1'], help='Render resulting policy or no', default=1)

    if mode == 'Polytype':
        parser.add_argument("exp", choices=['MLE', 'VE'], help="how to train the model", default='VE')
        parser.add_argument("num_policies", help="number of polcies to consider", type=int, default=30)
        parser.add_argument("-e", "--env", choices = ['C','F'], help="Catch or FourRooms", default='C' )
        
    args = parser.parse_args()

    return args