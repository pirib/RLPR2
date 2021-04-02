# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:21 2021

@author: babay
"""

import grid.py

import copy

class MCTS:
    
    
    # Constructor
    # episodes - total number of training episodes to run
    # num_rollouts - number of rollouts in a search 
    def __init__( self, board_size, episodes, num_rollouts):
        self.board_size = board_size
        self.episodes = episodes
        self.num_rollouts = num_rollouts
        
        
    # Execute a roullout search - returns a reward from the simulation
    # grid - the grid class that is currently in use. Will make a deep copy, and simulate the game on the copy.
    # policy - the policy that the rollout should use for simulations
    def rollout(self, grid, policy):
        state = copy.deepcopy(grid)
        
        
        
        