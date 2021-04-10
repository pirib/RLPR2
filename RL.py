# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:02:13 2021

@author: babay
"""

# In-house stuff
import MCTS as mt
import ANET as an
import grid

# For general funciton
import time

class RL():

    # Constructor
    def __init__(self, board_size, episodes, num_search_games, rollout_policy, grate, M, nn_layers, nn_optimizer):
        
        # 1. M is the save interval for ANET parameters
        
        # 2. Replay Buffer will store the potential training cases for the ANET
        self.RBUF = []
        
        # 3. Initialize the Neural Network (adding the input layer which is the square of the )
        self.ANET = an.ANET( [board_size**2] + nn_layers , nn_optimizer)
        
        # 4. Start working through episodes/epochs
        for e in range(episodes):
            
            # a. Initialize the actual game board
            board = grid.Grid(board_size)    

            # b. The initial board state
            # s_init = board.get_state()

            # c. Initialize the MCTS
            mcts = mt.MCTS(     anet = self.ANET,
                                board_size = board_size, 
                                episodes = episodes, 
                                num_search_games = num_search_games, 
                                rollout_policy = rollout_policy,
                                grate = grate)
            
            # d. While the board is not in terminal state
            while not board.is_terminal()[0]:

                # Start running mcts with the root (does so by the default). Uses ANET by default. Rollouts are done num_search_games times. 
                mcts.run()
                     
                # e. Training ANET from the 
                self.RBUF.append( ( mcts.root ,   ) )
            
            # f. Save the parameters of the NN for the evaluation
            if e % M == 0:
                self.ANET.save_NN(e)

    
    
start_time = time.time()

rl = RL(
        board_size = 4, 
        episodes = 500, 
        num_search_games = 300, 
        rollout_policy = "r", 
        grate = 0.2, 
        M = 200, 
        
        nn_layers = [ 4, "sigmoid", 4, "sigmoid", 16], 
        nn_optimizer = "SGD"
)




print("--- %s seconds ---" % (time.time() - start_time))























