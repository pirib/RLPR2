# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:12:32 2021

@author: babay
"""

# In-house stuff
import ANET as an
import grid
import h

import random


# Load the ANET

class TOPP():


    # M - the jump between the trianing episodes the NN took
    # N - number of participants
    # G - number of games in series
    def __init__(self, M, N, G, board_size, path):
        
        # [p1 p2] : win/loss ratio
        self.results = {}
        self.anets = []
        
        # Initialize the anets into a list and load the models
        for i in range(N):
            self.anets.append( an.ANET() )
            self.anets[i].load( str(i*M) , "./" + path )
       
        self.play(M, N, G, board_size)


    # Play the tournament
    def play(self, M, N, G, board_size):
        for anet1 in self.anets:
            for anet2 in [a for a in self.anets if a != anet1]:
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
                    key = "an" + str(self.anets.index(anet1)) + " vs an" + str(self.anets.index(anet2)) 
                    
                    # Add the results into the dictionary 
                    if play.is_terminal()[1] == 1:
                        if key in self.results: 
                            self.results[key][0] += 1                
                        else: 
                            self.results[key] = [1,0]
                    else:
                        if key in self.results: 
                            self.results[key][1] += 1                
                        else: 
                            self.results[key] = [0, 1]

    # Prints the results of the tounament
    def print_results(self):
            for x in self.results.keys():
                print(f"{x}: {self.results[x]}")

        

# Initialize the TOPP
topp = TOPP( M = 50, 
             N = 5, 
             G = 5, 
             board_size = 4,
             path = "4x4constgrate")

# Print the results
topp.print_results()



