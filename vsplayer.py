# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:32:20 2021

@author: babay
"""

# Not part of delivery - allows to play versus the AI


# In-house stuff
import ANET as an
import grid
import h

import random

class play():
    
    def __init__(self, board_size, ai_lvl, path, player):
        
        # Load the ANET
        anet = an.ANET()
        
        anet.load(ai_lvl, path)
        
        # Create the board
        play = grid.Grid(board_size)
        play.print_grid()
        
        # Playing until the board is in terminal state
        while not play.is_terminal()[0]:
        
            # Change 2 to 1 if you want to go first
            if play.get_player() == player:
        
                # Play the game, pausing every time it is player's turn to play
                player_input = input("Player turn: ")
                move = (int(player_input[1]), int(player_input[0]) )
        
            else:
        
                # Get the probability distribution from ANET
                pd = anet.policy( str(play.get_player()) + play.get_state() )
                
                # Pick the best move
                move = play.get_coor( pd.index( h.argmax( pd ))) 

            
            # Make the move and print the grid
            play.make_move(move)
            play.print_grid()
            
        # Printing the winner
        print("Player " + str(1 if play.get_player() == 2 else 2 ) + " won!" )


# Initialize
t = play(board_size = 6, 
         ai_lvl = 210, 
         path = "6x6", 
         player = 2)






