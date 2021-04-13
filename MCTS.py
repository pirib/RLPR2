# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:20:21 2021

@author: babay
"""

# In-house stuff
import grid
import h

# For general funciton
import random
from math import log10

class MCTS:
     
    # Constructor
    # anet - the neural network that should be used for rollout selection
    # episodes - total number of training episodes to run
    # grate - greed rate in the tree selection policy
    # num_rollouts - number of rollouts in the simulation search 
    def __init__( self, anet, board_size, grate, rollout_policy = "n", num_search_games = 1):
        self.anet = anet
        self.board_size = board_size
        self.num_search_games = num_search_games
        self.policy = rollout_policy
        self.grate = grate

        # Set the root node 
        # The root of the MCTS - constructor initiliazes the parent to None, while state is a string with all 0 for board_size**2
        self.root = snode("".join("0" for s in range(board_size**2)) , None )
        
        # The board this MCTS's root is currently in    
        self.bb = grid.Grid(board_size)
        
    # Run the MCTS
    def run(self):
        # The 4 procedures 
        
        # 1. Selection
        snode, anode = self.selection() 
        # 2. Expansion
        esnode = self.expansion(snode, anode)
        # 3. Simulation
        reward, sel_anode = self.simulation(esnode, self.policy)
        # 4. Backup
        self.backup(sel_anode, reward)


    
    # The 4 horsemen of MCTS
    
    # e-greedy tree traversal - picks the snode+anode that should be expaned further (e.g. pick a state/action pair). 
    # Returns snode and anode for expansion
    def selection(self):
    
        # Iterates through the tree, starting at the root, with action leading to it
        def iterate(snode):
            
            # Select an action using tree policy
            anode = self.tree_policy(snode, self.grate)
            
            # If the anode is None, it means that there are no actions available from the snode
            if anode == None:
                return snode, None 
                
            # If the action has not been expanded yet, we are done, return the state-action pair we are after
            if anode.child == None:
                return snode, anode

            # If there is a child node attached, then we just jump further
            else:
                return iterate(anode.child)

        # Recrusively calls itself, until the action node that is going to be expanded, is returned
        return iterate(self.root)
      

    # Expands the given state node using the action provided. Returns the expanded state node
    def expansion(self, state_node, action_node):
        
        # If action node is None, we are in the final state, there is no need for expansation
        if action_node == None:
            return state_node
        
        # Make a board from the anode's parent state
        temp_board = grid.create_board(state_node.state)
        
        # Make a move based on the anode
        temp_board.make_move(action_node.action)
        
        # Create a state node, add possible action nodes and assign the new snode as a child of the anode
        action_node.child = snode( temp_board.get_state() , action_node)
    
        # Return it
        return action_node.child
    
    
    # From the chosen state node, tree policy selects an action from which the rollout simulation is run
    # Returns the reward and the anode that was chosen (for backup)
    def simulation(self, snode, policy = "n"):

        # Execute a roullout search - returns a reward from the terminal state
        # anode - the anode from which the rollouts will be made
        # policy - the policy that the rollout should use for simulations
        def rollout(anode, policy):
            
            # Make the board from the anode's parent's state
            board = grid.create_board(anode.parent.state)
            
            # Make a move based on the anode selected
            board.make_move(anode.action)
                            
            # Make moves until the game is in terminal state
            while not board.is_terminal()[0]:

                move = None
                
                # Random policy, e.g. moves are selected randomly
                if policy == "r":   
                    move = random.choice(board.get_available_actions())
                
                # ANET should predict the next move
                elif policy == "n":
                    
                    if random.random() > self.grate:
                        # Pick a move randomly
                        move = random.choice(board.get_available_actions())
                    
                    else:
                        # Ask anet to select the move for the next state
                        pd = self.anet.policy( str(board.get_player()) + board.get_state())
                        # Get the index of the highest PD value, then its coordinate
                        move = board.get_coor( pd.index(h.argmax( pd )) )

                # Raise the exception if wrong rollout policy has been chosen
                else:
                    raise Exception("MCTS does not support policy named " + policy)

                # Make the move selected by the rollout policy
                board.make_move(move)
                
            # Return the reward of the terminal state
            return board.get_reward()
        # ============================================

        # Select an action node to run simulations with
        anode = self.tree_policy(snode, self.grate)
        
        # If anode is None - then the board is in terminal state, game has finished.
        if anode == None:
            # We just need to return its reward (there is no need for Rollouts)
            board = grid.create_board(snode.state)
            return board.get_reward(), snode.parent
                
        # Return the average reward from all the rollouts and the anode that was chosen
        return rollout(anode, policy), anode
    
    
    
    # Backpropagate the reward information down to the root
    def backup(self, action_node, reward):
        
        # Iterative backpropagation
        def bp(action_node):
            
            # Update the visit count for the action_node and its parent and the new value for the state/action value
            action_node.update_value_visit(reward)
            action_node.parent.update_visits()
            
            # If we haven't reached the root yet continue propagating upwards
            if not action_node.parent.parent == None:
                # Backpropagate to the parents parents
                bp(action_node.parent.parent)
        
        # Call the function iteratively
        bp(action_node)
        
    
    
    # The tree policy - e-greedy policy, expects a state node, returns an action node.
    # The choice the tree makes is based on the VALUES of the actions
    def tree_policy(self, snode, grate):

        # Upper Confidence Bound to encourage exploration
        def UCT(state_visits, action_visits):
            return ( log10(state_visits) / (1 + action_visits) )**0.5 if state_visits != 0 else 0            

        # If snode received is None, something went wrong
        if h.is_empty(snode) :
            raise Exception("Tree policy received a None instead of a snode")
        
        # If the board is already in the terminal state, e.g. no actions are available, return None
        elif h.is_empty(snode.actions):
            return None
  
        # With grate probability explore
        if random.random() > self.grate:
            # Pick a move randomly
            anode = random.choice(snode.actions)

        # Else, exploit, based on the values of the action, and the player's turn it currently is
        else:
            # Get the state info
            board = grid.Grid( self.board_size )
            board.set_from_state(snode.state)
            
            # Make a move depending whose turn it is
            if board.get_player() == 1:
                # Select the action which gives the highest values 
                anode = h.argmax(snode.actions, (lambda a : a.value + UCT( a.parent.visits , a.visits ))  )
            else:
                # Select the action which gives the lowest values 
                anode = h.argmin(snode.actions, (lambda a : a.value - UCT( a.parent.visits , a.visits ))  )
        
        # Return the selected action
        return anode
    


# Node classes =============================================================================
        
# State node
class snode():
         
    # Constructor
    def __init__(self, state, parent):
        # State of the board at the current node - given as a string of 0,1,2, same as compact state representation.
        self.state = state
        # The action node from which this state is reached
        self.parent = parent
        # Visits this node has received
        self.visits = 0
        # A list of action nodes - e.g. all the child anodes from this state
        self.actions = []
        # Generate the actions that are avvailable from this state node
        self.gen_actions(self)
        
        
    # Generate actions, e.g. child action nodes
    def gen_actions(self, parent):
        
        # Make a board based on the parent's state
        board = grid.create_board(parent.state)
        
        # Generate actions
        aa = board.get_available_actions()
        
        # If there are no available actions, this is a terminal node
        if aa == None:
            self.actions = None
        # Else, add all possible moves as action nodes
        else:
            for a in board.get_available_actions():
                self.actions.append( anode( a, self ) )


    # Get the array consisting of the visit counts. 0 is set for the un available actions
    def get_visits(self):
        
        # Initialize a board for the same state
        board = grid.Grid( int(len(self.state)**0.5) )
        board.set_from_state(self.state)
        
        # Empty list that we are going to fill up
        visits = []
        
        # Get all the nodes
        b_states = board.get_state(False)

        # Match is with legal actions
        for b in b_states:
            found = False
            i = None
            for ai in range(len(self.actions)):
                # If there is such an ai that 
                if b == self.actions[ai].action: 
                    found = True
                    i = ai 
            
            # Set to 0 if the action is not legal
            if found: visits.append(self.actions[i].visits)
            else: visits.append(0)
        
        # Return the complete list
        return visits
    
    # Addds 1 to the visits
    def update_visits(self):
        self.visits = self.visits + 1

    
# Action node
class anode():
    
    # Constructor
    def __init__(self, action , parent):
        # The state node from which this action is taken
        self.parent = parent
        # A tuple indicating the move
        self.action = action
        # The value of the action. Together with parent this consitutes an state-action pair
        self.value = 0
        # Number of time this anode has been visited
        self.visits = 0
        # The state node this action leads to
        self.child = None

    # Updates the value of the action node and its visited counter
    def update_value_visit(self, new_value):
        tv = self.value * self.visits
        self.visits = self.visits + 1
        self.value = (tv + new_value) / self.visits 














    