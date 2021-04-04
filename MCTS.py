# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:21 2021

@author: babay
"""

import grid
import h

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
        def iterate(root):
            action = None
                       
            # With grate probability explore
            if random.random() <= self.grate:
                # Pick a move randomly
                action = random.choice(root.actions)
                                
            # Else, exploit, based on the values of the action
            else:
                action = h.argmax(root.actions, lambda a : a.value)
                
            # If the action has no child we are done, this is our new leaf node, the expansion happens next
            if not action.child:
                return action

            # If there is a child node attached, then we just jump further
            else:
                return iterate(action.child)

        # Recrusively calls, until 
        return iterate(self.root)
    
    
    # Expands the given action node, by adding a 
    def expansion(self, anode):
        
        # Create a state node and assign it as a child of the anode
        anode.child = snode( "" , anode)
    
    
    def simulation(self):
        pass
    
    def backup(self):
        pass
    
        
    # Execute a roullout search - returns a reward from the simulation
    # board - the grid class that is currently in use. Will make a deep copy, and simulate the game on the copy.
    # policy - the policy that the rollout should use for simulations
    def rollout(self, board, policy = "r"):
        
        # Make a copy of the board for the rollout
        state = copy.deepcopy(board)
        
        # Random policy, e.g. moves are selected randomly
        if policy == "r":
            
            # Make moves until the game is in terminal state
            while not state.is_terminal()[0]:
                move = random.choice(state.get_available_actions())
                state.make_move(move , 1)
                
            # Return the reward of the terminal state
            return state.get_reward()
        
        elif policy == "n":
            return
    
        
        
# State node
# TODO gen_actions needs to be done next
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
    def gen_actions(self):
        pass
   
    
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
    def __init__(self):
        pass
















    