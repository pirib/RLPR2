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
            
            # a. Initialize an empty board
            board = grid.Grid(board_size)    

            # b. The initial board state
            # s_init = board.get_state()

            # c. Initialize the MCTS
            self.mcts = mt.MCTS(    anet = self.ANET,
                                    board_size = board_size, 
                                    num_search_games = num_search_games, 
                                    rollout_policy = rollout_policy,
                                    grate = grate)
            
            
            # d. Run it while the board is not in terminal state
            while not self.mcts.board.is_terminal()[0]:
                                                
                # Testing grounds - running until time expires instead of num_search_games
                # t_end = time.time() + 2
                # time.time() < t_end

                # Start running mcts with the root (does so by the default). Uses ANET by default. Rollouts are done num_search_games times.                 
                # Run MCTS for num_search_games times
                for sg in range(num_search_games): 
                    self.mcts.run()

                # Adding training data for the ANET to learn from 
                self.RBUF.append( ( self.mcts.root.state , self.mcts.root.get_visits()  ) )

                # Use tree policy (full greedy choice with the highest action), move the board to the new state, and make the new successor state the root
                chosen_an = self.mcts.tree_policy(self.mcts.root, grate = 1)
                self.mcts.board.make_move(chosen_an.action)
                self.mcts.root = chosen_an.child


            # Now that we are collecting a database of cool ass moves, we need to train our network with a random minibatch from there
            # e. Train ANET from the RBUF            
            for i in range(int(len(self.RBUF) / 5 ) ):
                # Pick a random training case and train the ANET
                case = random.choice( self.RBUF ) 
                self.ANET.train( case[0], case[1]  )
                
                
            # f. Save the parameters of the NN for the evaluation
            if e % M == 0:
                self.ANET.save(e)
                
            
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
        board_size = 3, 
        episodes = 200, 
        num_search_games = 400, 
        rollout_policy = "n", 
        grate = 0.2, 
        M = 50, 
        
        nn_layers = [ 3, "sigmoid", 3, "sigmoid"], 
        nn_optimizer = "SGD"
)




print("--- %s seconds ---" % (time.time() - start_time))























