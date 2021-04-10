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

class MCTS:
     
    # Constructor
    # anet - the neural network that should be used for rollout selection
    # episodes - total number of training episodes to run
    # grate - greed rate in the tree selection policy
    # num_rollouts - number of rollouts in the simulation search 
    def __init__( self, anet, board_size, episodes, grate, rollout_policy = "n", num_search_games = 1):
        self.anet = anet
        self.board_size = board_size
        self.episodes = episodes
        self.num_search_games = num_search_games
        self.policy = rollout_policy
        self.grate = grate

        # Set the root node 
        # The root of the MCTS - constructor initiliazes the parent to None, while state is a string with all 0 for board_size**2
        self.root = snode("".join("0" for s in range(board_size**2)) , None )
        
        
    # Run the MCTS 
    def run(self):
                    
        anode = self.selection()
        snode = self.expansion(anode)
        reward, sel_anode = self.simulation(snode, self.policy)
        self.backup(sel_anode, reward)        
        
        
    # The 4 horseman of MCTS
    
    # e-greedy tree traversal - picks the node that should be expaned further (e.g. pick a state/action pair). 
    # Returns an action node that is chosen for expansion
    def selection(self):
    
        # Iterates through the tree, starting at the root, with action leading to it
        def iterate(snode):
            
            # 1. Select an action using tree policy
            anode = self.tree_policy(snode)
                       
            # 2. See what the action selection led us to
                
            # If the action has no child we are done, this is our new leaf node, the expansion happens next
            if not anode:
                return snode.parent
            
            if not anode.child:
                return anode

            # If there is a child node attached, then we just jump further
            else:
                return iterate(anode.child)

        # Recrusively calls itself, until the action node that is going to be expanded, is returned
        return iterate(self.root)
    
    
    # Expands the given action node, by adding a child state node to the action node, and expanding to the possible actions
    # Returns snode, the generated child of the anode
    def expansion(self, anode):
        
        # Make a board from the anode's parent state
        board = grid.create_board(anode.parent.state)
        
        # Make a move based on the anode
        board.make_move(anode.action)
        
        # Create a state node and assign it as a child of the anode
        anode.child = snode( board.get_state() , anode)
    
        # Return it
        return anode.child
    
    
    # From the chosen state node, tree policy selects an action from which the rollout simulations are ran
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
                    
                    # Ask anet to predict move for the next state
                    pd = self.anet.policy(board.get_state())
                    
                    # Get the index of the highest PD value, then its coordinate
                    move = board.get_coor( pd.index(h.argmax( pd )))
                
                else:
                    raise Exception("MCTS does not support policy named " + policy)
                    
                board.make_move(move)
                
            # Return the reward of the terminal state
            return board.get_reward()


        # Select an action node to run simulations with
        anode = self.tree_policy(snode)
        
        # If anode is None
        if not anode:
            # Then the board is in terminal state, we just need to return its reward (there is no need for Rollouts)
            board = grid.create_board(snode.state)
            return board.get_reward(), snode.parent
                
        # Calculate the average rewards from all the rollouts done
        tr = 0
        for r in range(self.num_search_games):
            tr += rollout(anode, policy)
        
        # Return the average reward from all the rollouts and the anode that was chosen
        return tr/self.num_search_games, anode
    
    
    # Backpropagate the reward information down to the root
    def backup(self, anode, reward):
        
        # Iterative backpropagation
        def bp(anode):
            
            # Update the visit count and the new value for the state/action value
            anode.visits += 1
            anode.update_value(reward)
            
            # If the anode's parent's parent is not None 
            # (e.g. we haven't reached the root yet), then continue propagating upwards
            if not anode.parent.parent == None:
                # Backpropagate to the parents parents
                bp(anode.parent.parent)
        
        # Call the function iteratively
        bp(anode)
        
    
    # The tree policy - e-greedy policy, expects a state node, returns an action node
    # The choice the tree makes is based on the VALUES of the actions rather than visit statistics
    def tree_policy(self, snode ):

        # If the board is already in the terminal state, e.g. no actions are available, return None
        if h.is_empty(snode.actions):
            return None
                           
        # With grate probability explore
        if random.random() <= self.grate:
            # Pick a move randomly
            anode = random.choice(snode.actions)
                            
        # Else, exploit, based on the values of the action, and the player's turn it currently is
        else:
            
            # Get the state info
            board = grid.Grid( self.board_size )
            board.set_from_state(snode.state)
                                    
            # Make a move depending whose turn it is
            if board.get_player() == 1:
                anode = h.argmax(snode.actions, lambda a : a.value)
            else:
                anode = h.argmin(snode.actions, lambda a : a.value)
            

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
        # A list of action nodes - e.g. all the child anodes from this state
        self.actions = []
        # Generate the actions that are avvailable from this state node
        self.gen_actions(self)
        
        
    # Generate actions, e.g. child action nodes
    def gen_actions(self, parent):
        
        # Make a board based on the parent's state
        board = grid.create_board(parent.state)
        
        # Generate actions
        for a in board.get_available_actions():
            self.actions.append( anode( self, a) )


    # Get the array consisting of the visit counts. 0 is set for the un available actions
    def get_visits(self):
        
        # Initialize a board for the same state
        board = grid.Grid( int(len(self.state)**0.5) )
        board.set_from_state(self.state)
        
        visits = []

        for n in board.get_state(False):
            for a in self.actions:
                if n == a.action:
                    visits.append( self.actions[self.actions.index(a)].visits )   
                    break
                else:
                    visits.append(0)
                    break
                
        return visits
    
# Action node
class anode():
    

    # Constructor
    def __init__(self, parent, action):
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


    def update_value(self, new_value):
        tv = self.value * self.visits
        self.value = (tv + new_value) / self.visits 














    