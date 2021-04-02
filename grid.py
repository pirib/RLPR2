# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:07:25 2021

@author: babay
"""

# In-house stuff
import node as N

# For general funciton
import random
import time

# For printing pretty stuff
import networkx as nx
import matplotlib.pyplot as plt 



class Grid():

    # Private
    grid = []
    
    # Potential neighbours set for each node - (row, col)
    offset = ( (-1,-1) , (0,-1) , (-1,0) , ( 0, 1) , ( 1, 0) , (1, 1) )

    # Methods
    
    # Grid:
    # Size is as defined in hex-board-games.pdf    
    def __init__( self, size):
        self.grid.clear()
        self.create(size)
        self.size = size
        
        
    # Destructor - also removes all the nodes.
    def __del__(self): 
        for row in self.grid:
            for node in row:
                del node
        del self
        
        
    # Creates the grid
    def create(self, size):
                    
        # Creating nodes
        for r in range(size):
            # Create a new empty row
            self.grid.append([])
                
            for c in range (size):
                # Make a new node and shove into the right row. The node is then accessible at grid[r][c].
                self.grid[r].append( N.Node(r,c,size) )

        # Setting nodes neighbours
        
        # For each node in the grid, check if there is a neighbour in one of the possible offset coordinates
        for row in self.grid:
            for node in row:
                for o in self.offset:
                    
                    # Potential neighbour row/col coordinates
                    p_n_r = node.row + o[0]
                    p_n_c = node.col + o[1]
                    
                    # As long as potential coordinates are not negative (since Python allows usage negative indexes...)
                    if p_n_c >= 0 and p_n_r >= 0:
                        
                        try:
                            node.neighbours.append( self.grid[p_n_r][p_n_c] )
                        except:
                            pass                 
        
                
    # Places a piece into the an empty spot as specified by coor tuple and player parameters
    # This function moves the grid into a new state
    def make_move(self, coor, player):
        if self.grid[coor[0]][coor[1]].piece == 0 :
            self.grid[coor[0]][coor[1]].piece = player
        else: 
            raise Exception("Player " + player + " attempted to make an illegal move!")


    # Returns the state representation. if compact == True then a string with 0 for empty, 1 and 2 for player 1 and 2 will be returned
    def get_state(self, compact = True):
        
        if compact:
            state = ""
            
            for row in self.grid:
                for node in row:
                    state = state + str(node.piece)        
                    
            return state

        else:
            
            state = []
            
            for row in self.grid:
                for node in row:
                    state.append( ( node.row, node.col ) )

            return tuple(state)


    # Returns a tuple of tuples with empty spots
    def get_available_actions(self):
        
        actions = []
        
        for row in self.grid:
            for node in row:
                if node.piece == 0:
                    actions.append( (node.row, node.col) )
                    
        return tuple(actions)
        
        
    # Depth-first search from an opposing side to another
    # Returns True if the state is terminal, as well as the player number who won the game
    # TODO return also the winning route?
    def is_terminal(self):

        def iterate(node,visited,player):
            
            for n in node.neighbours:
                if n not in visited and player == n.piece:
                    # Visit the next node
                    visited.append(n)

                    # Return True if reached the other border
                    
                    if player == 1:
                        for nb in [self.grid[x][y] for x,y in [ (self.size-1, i) for i in range(self.size) ] ]:   
                            if nb == n:                                
                                return True
                    
                    else:
                        for nb in [self.grid[x][y] for x,y in [ (i, self.size-1) for i in range(self.size) ] ]:   
                            if nb == n:
                                return True
                            
                    # Else, iterate further
                    if iterate(n, visited, player):
                        return True
                     
                               
        # Checking for each possible player
        for player in range(1,3):
            
            # Loop through the nodes that are in the starting border (e.g. max number of nodes in the starting/winning border == size of the board)
            for i in range(self.size):
                
                # Only looping through the nodes that have the pieces belonging to the current player                
                if player == 1:
                    if self.grid[0][i].piece == player and iterate(self.grid[0][i], [], player):
                        return True, 1
                
                else:
                    if self.grid[i][0].piece == player and iterate(self.grid[i][0], [], player):
                        return True, 2
                    
                    
        # If none of the iterations reached the other border, return False
        return False
    
    
    # TODO
    # The reward should depend on the player playing ?
    def get_reward(self):
            return 0
    
    
    # Prints out a pretty looking 
    # TODO show the winning route ?
    def print_grid(self):
        
        # The new graph for printing
        G = nx.Graph()

        # The colors and labels used in the drawing
        color_map = []
        labels = {}

        # Iterate through all the nodes, and add them as.. nodes
        for row in self.grid:
            for n in row:
                G.add_node(n) 
                if n.piece == 1:
                    color_map.append('red') 
                elif n.piece == 2: 
                    color_map.append('blue')
                else:
                    color_map.append('white') 
                
                labels[n] = [n.row, n.col]
                
        # Iterate through each neighbour of the node, and add the edges in between. Networkx ignores already existing edges, which is nice
        for row in self.grid:
            for node in row:
                for n in node.neighbours:
                    G.add_edge(node, n)
        
        # Draws the nodes 
    
        nx.draw(G, labels, labels=labels, node_color=color_map)
        plt.pause(0.001)


    # End of the Class =============================================================
    


# Plays a game randmoly picking available actions

play = Grid(4)

player = 1    
while ( True ):
    temp = random.choice(  play.get_available_actions()  )
    play.make_move( temp ,  player )
    #time.sleep(0.5)
    play.print_grid()
    print("Player " + str(player) + " places a piece in " + str(temp))
    player = 2 if player == 1 else 1
    
    results = play.is_terminal()
    if type(results) != bool and results[0]:
        print("Player " + str(results[1]) + " won the game!")
        break
    















