# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:02:13 2021

@author: babay
"""

# In-house stuff
import MCTS as mt
import ANET
import grid



class RL():
 
    mcts = None   
 
    def __init__(self, board_size, grate, episodes, num_rollouts):
        self.run(board_size, grate, episodes, num_rollouts)
    
    
    def run(self, board_size, grate, episodes, num_rollouts):
        self.mcts = mt.MCTS(board_size, grate, episodes, num_rollouts)
        self.mcts.run()
    
    
    
rl = RL(4, 0.2, 100, 100)