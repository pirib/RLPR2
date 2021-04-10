# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:02:13 2021

@author: babay
"""

# In-house stuff
import MCTS as mt
import ANET as an
import grid
import h

# For general funciton
import time
import random


class RL():

    # Constructor
    def __init__(self, board_size, episodes, num_search_games, rollout_policy, grate, M, nn_layers, nn_optimizer):
        
        self.board_size = board_size
        
        # 1. M is the save interval for ANET parameters
        
        # 2. Replay Buffer will store the potential training cases for the ANET
        self.RBUF = []
        
        # 3. Initialize the Neural Network (adding the input layer which is the square of the )
        self.ANET = an.ANET( [board_size**2] + nn_layers + [board_size**2] , nn_optimizer)
        
        # 4. Start working through episodes/epochs
        for e in range(episodes):
            
            # a. Initialize the actual game board
            board = grid.Grid(board_size)    

            # b. The initial board state
            # s_init = board.get_state()

            # c. Initialize the MCTS
            self.mcts = mt.MCTS(     anet = self.ANET,
                                board_size = board_size, 
                                episodes = episodes, 
                                num_search_games = num_search_games, 
                                rollout_policy = rollout_policy,
                                grate = grate)
            
            # d. Run it while the time remains
            t_end = time.time() + 2
            while time.time() < t_end:
                
                # Start running mcts with the root (does so by the default). Uses ANET by default. Rollouts are done num_search_games times. 
                self.mcts.run()
                
                # e. Training ANET from the 
                self.RBUF.append( ( self.mcts.root.state , self.mcts.root.get_visits()  ) )
            
            
            # e. Train ANET from the RBUF            
            for i in range(int(len(self.RBUF) / 5 ) ):
                # Pick a random training case
                case = random.choice( self.RBUF ) 
                
                self.ANET.train( case[0], case[1]  )
                
                
            # f. Save the parameters of the NN for the evaluation
            if e % M == 0:
                pass
                # self.ANET.save_NN(e)
            
            
    # Play using ANET
    def play(self, print_grid = True):
        
        play = grid.Grid(self.board_size)
        
        # Play until in the final state
        while not play.is_terminal()[0]:
            # Get the probability distribution from ANET
            pd = self.ANET.policy(play.get_state())
            
            move = play.get_coor( pd.index(h.argmax( pd )))
            
            play.make_move(move)

            # Print the grid
            if print_grid: play.print_grid()
        
        

start_time = time.time()

rl = RL(
        board_size = 5, 
        episodes = 100, 
        num_search_games = 100, 
        rollout_policy = "r", 
        grate = 0.2, 
        M = 200, 
        
        nn_layers = [ 4, "sigmoid", 4, "sigmoid"], 
        nn_optimizer = "SGD"
)




print("--- %s seconds ---" % (time.time() - start_time))























