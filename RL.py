# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:02:13 2021

@author: babay
"""

# In-house stuff
import MCTS as mt
import ANET
import grid

# For general funciton
import time

class RL():
 
    mcts = None   
 
    def __init__(self, board_size, episodes, num_rollouts, grate):
        self.run(board_size = board_size, 
                 episodes = episodes, 
                 num_rollouts = num_rollouts, 
                 grate = grate)
    
    
    def run(self, board_size, episodes, num_rollouts, grate):
        self.mcts = mt.MCTS(board_size, episodes, num_rollouts, grate)
        self.mcts.run()
    
    
start_time = time.time()
rl = RL(4, 1000, 1000, 0.2)
print("--- %s seconds ---" % (time.time() - start_time))
