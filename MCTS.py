# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:21 2021

@author: babay
"""

# In-house stuff
import grid
import h

# For general funciton
import copy
import random

class MCTS:
    
    # The root of the MCTS - constructor initiliazes the parent to None, while state is a string with all 0 for board_size**2
    root = None
    
    # Constructor
    # episodes - total number of training episodes to run
    # grate - greed rate in the tree selection policy
    # num_rollouts - number of rollouts in a search 
    def __init__( self, board_size, grate, episodes, num_rollouts):
        self.board_size = board_size
        self.grate = grate
        self.episodes = episodes
        self.num_rollouts = num_rollouts
        self.root = snode("".join("0" for s in range(board_size**2)) , None )
        
        
    # The 4 horseman of MCTS
    
    # Traverse the tree and pick the node that should be expaned further (e.g. pick a state/action pair)
    # e-greedy policy is used
    def selection(self):
    
        # Iterates through the tree, starting at the root, with action leading to it
        def iterate(snode):
            
            # 1. Select an action using tree policy
            anode = self.tree_policy(snode)
                       
            # 2. See what the action selection led us to
                
            # If the action has no child we are done, this is our new leaf node, the expansion happens next
            if not anode.child:
                return anode

            # If there is a child node attached, then we just jump further
            else:
                return iterate(anode.child)

        # Recrusively calls, until 
        return iterate(self.root)
    
    
    # Expands the given action node, by adding a child state node to the action node, and expanding to the possible actions
    def expansion(self, anode):
        
        # Make a board from the anode's parent state
        board = grid.create_board(anode.parent.state)
        
        # Make a move based on the anode
        board.make_move(anode.action)
        
        # Create a state node and assign it as a child of the anode
        anode.child = snode( board.get_state() , anode)
    
    
    # Simulate
    def simulation(self, snode):

        # Execute a roullout search - returns a reward from the simulation
        # anode - the anode from which the rollouts will be made
        # policy - the policy that the rollout should use for simulations
        def rollout(self, anode, policy = "r"):
            
            # Make the board from the anode's parent's state
            board = grid.create_board(anode.parent.state)
            
            # Make a move based on the anode selected
            board.make_move(anode.action)
            
            # Random policy, e.g. moves are selected randomly
            if policy == "r":
                
                # Make moves until the game is in terminal state
                while not board.is_terminal()[0]:
                    move = random.choice(board.get_available_actions())
                    board.make_move(move)
                    
                # Return the reward of the terminal state
                return board.get_reward()
            
            elif policy == "n":
                raise Exception("Policy other than random rollout have not been done yet")


        # Select an action node to run simulations with
        anode = self.tree_policy(snode)
        
        return self.rollout(anode)
    
    
    def backup(self):
        pass
    
        
    
    # The tree policy - e-greedy policy, expects a state node, returns an action node
    def tree_policy(self, snode):
                           
        # With grate probability explore
        if random.random() <= self.grate:
            # Pick a move randomly
            anode = random.choice(snode.actions)
                            
        # Else, exploit, based on the values of the action
        else:
            anode = h.argmax(snode.actions, lambda a : a.value)
        
        return anode
    

        
# State node
class snode():
        
    # The action node from which this state is reached
    parent = None
    
    # State of the board at the current node - given as a string of 0,1,2, same as compact state representation.
    state = ""
    
    # A list of action nodes - e.g. all the child anodes from this state
    actions = []
    
    # Constructor
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.gen_actions()
        
    # Generate actions, e.g. child nodes
    def gen_actions(self, parent):
        
        # Make a board based on the parent's state
        board = grid.create_board(parent.state)
        
        # Generate actions
        for a in board.get_available_actions():
            self.actions.append( anode( self, a) )

   
    
# Action node
class anode():
    
    # The state node from which this action is taken
    parent = None
    
    # The value of the action. Together with parent this consitutes an state-action pair
    value = 0
    
    # A tuple indicating the move
    action = None
    
    # The state node this action leads to
    child = None

    # Constructor
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
















    