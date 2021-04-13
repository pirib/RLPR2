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
        self.RBUF = {}
        
        # 3. Initialize the Neural Network (adding the input layer which is the square of the )
        self.ANET = an.ANET( [board_size**2] + nn_layers + [board_size**2] , nn_optimizer)
        
        # 4. Start working through episodes/epochs
        for e in range(episodes):
            
            # Just so we are aware of the training process
            print("episode " + str(e))
            
        
            # a. b. c. Initialize the MCTS
            self.mcts = mt.MCTS(    anet = self.ANET,
                                    board_size = board_size, 
                                    num_search_games = num_search_games, 
                                    rollout_policy = rollout_policy,
                                    grate = grate)

            # Grate drops every episode
            grate *= 0.99

            # d. Run it while the board is not in terminal state
            while not self.mcts.bb.is_terminal()[0]:
                
    
                # Start running mcts with the root (does so by the default). Uses ANET by default. Rollouts are done num_search_games times.                 
                # Run MCTS for num_search_games times
                for sg in range(num_search_games): 
                    self.mcts.run()

                # Adding training data for the ANET to learn from 
                self.RBUF[str(self.mcts.bb.get_player()) + self.mcts.root.state ] = self.mcts.root.get_visits() 

                #=========================== TESTING GROUNDS
                #print(self.mcts.root.state)
                #print(self.mcts.root.get_visits())
                #print([a.value for a in self.mcts.root.actions])
                #=========================== TESTING GROUNDS

                # Use tree policy (full greedy choice with the highest value of the action), move the board to the new state, and make the new successor state the root
                chosen_an = self.mcts.tree_policy(self.mcts.root, grate = 1)
                self.mcts.bb.make_move(chosen_an.action)
                self.mcts.root = chosen_an.child


            #print(self.RBUF)
            #print()
            # Now that we are collecting a database of cool ass moves, we need to train our network with a random minibatch from there
            # e. Train ANET from the RBUF                
            for i in range(int(len(self.RBUF) * 0.3)):
                # Pick a random training case and train the ANET
                case = random.choice( list(self.RBUF.items()) )
                self.ANET.train( case[0], case[1] )
                
            # f. Save the parameters of the NN for the evaluation
            if e % M == 0:
                self.ANET.save(e)
                
            
    # Play using ANET
    def play(self, print_grid = True):
        
        play = grid.Grid(self.board_size)
        
        # Play until in the final state
        while not play.is_terminal()[0]:
            
            # Get the probability distribution from ANET
            pd = self.ANET.policy( str(play.get_player()) + play.get_state() )
            
            move = play.get_coor( pd.index(h.argmax( pd )))
            
            play.make_move(move)

            # Print the grid
            if print_grid: play.print_grid()
        
        

start_time = time.time()

rl = RL(
        board_size = 3, 
        episodes = 301, 
        num_search_games = 1000,
        rollout_policy = "n", 
        grate = 0.99, 
        M = 50, 
        
        nn_layers = [128, "relu"], 
        nn_optimizer = "Adam"
)



print("--- %s seconds ---" % (time.time() - start_time))























