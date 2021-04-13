# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:12:32 2021

@author: babay
"""

# In-house stuff

import ANET as an
import grid
import h


# Load the ANET

class TOPP():

    # n - number of participants
    # M - the jump between the trianing episodes the NN took
    def __init__(self, M, N, G, board_size):
        
        # [p1 p2] : win/loss ratio
        self.results = {}
        self.anets = []
        
        # Initialize the anets into a list and load the models
        for i in range(N):
            self.anets.append( an.ANET() )
            self.anets[i].load("./test/m" + str(i*M))
            self.play(M, N, G, board_size)


    def play(self, M, N, G, board_size):
        for anet1 in self.anets:
            for anet2 in self.anets:
                for i in range(G):
                    play = grid.Grid(board_size)
    
                    # Playing until the board is in terminal state. Anet1 is always the first player 
                    # This ensures that all networks will have a chance to play as first player
                    while not play.is_terminal()[0]:
                        
                        # Let the player whose turn it currently is to make the move
                        an = anet1 if play.get_player() == 1 else anet2
                            
                        # Get the probability distribution from ANET
                        pd = an.policy( str(play.get_player()) + play.get_state() )
                        
                        # Make the best move
                        play.make_move(play.get_coor( pd.index(h.argmax( pd ))) )
                    
                    # Save the results of the game
                    key = "p" + str(self.anets.index(anet1)) + " vs p" + str(self.anets.index(anet2)) 
                    
                    if play.is_terminal()[1] == 1:
                        if key in self.results: 
                            self.results[key] = self.results[key] + 1                
                        else: 
                            self.results[key] = 1
        

topp = TOPP(50, 5, 10, 3)


print(topp.results)



