# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:14:05 2021

@author: babay
"""

import tensorflow as tf
import numpy as np
import random
import grid
import scipy

tf.config.optimizer.set_jit(True)

class ANET():
    

    # Constructor
    # layers - a list with number of nodes in the layers, alternating with activation functions. The first value is the input shape only. 
    # ^ Activating functions accepted - "linear" , "sigmoid" , "tanh", "relu" 
    # optimizer - the optimizer used. 
    # ^ Following are accepted - "Adagrad" , "SGD" , "RMSprop" , "Adam" 
    def __init__( self,  layers = None, optimizer = "Adam"):
        # The model will be accessible directly
        self.model = None
        # Create the network
        if layers:
            self.create_network(layers, optimizer = optimizer)
    
    
    
    # Creates the network with specified parameters 
    def create_network(self, layers, optimizer):

        # Create the model
        self.model = tf.keras.Sequential()

        # Adding the input layer
        self.model.add(tf.keras.layers.InputLayer(input_shape = (layers[0]+1, ) ))
        #self.model.add( tf.keras.layers.Dense( units = layers[0]+1, activation = "sigmoid" ) )

        # Adding layers with number of nodes as specified in layers argument
        for i in range(len(layers[1:])):

            # Care only for the odd values, even are for the actionvation function
            if i % 2 != 0:
                self.model.add( tf.keras.layers.Dense( units = layers[i], activation = layers[i+1]) )
    
        # Add the output layer with softmax
        self.model.add(tf.keras.layers.Dense( units = layers[-1], activation='softmax' ) )
    
        # Compile the model
        self.model.compile(optimizer = optimizer, loss = 'categorical_crossentropy')
        
        
        
    # Train the network
    def train(self, state, visit_counts, e = 8):
        #print()
        #print(state)
        
        # Normalize visit_counts
        #visit_counts = [i/sum(visit_counts) for i in visit_counts ]
        
        # Softmax visit counts
        visit_counts = scipy.special.softmax(visit_counts)
        
        #print(visit_counts)
        # Setting the dimensions of the training data
        x = np.array(self.state_to_arr(state))
        x = np.expand_dims(x,0)
        #print(x)
        
        # Seeting the dimensions of the ouptut 
        y = np.array(visit_counts)
        y = np.expand_dims(y,0)
        #print(y)

        self.model.fit( x, y, epochs = e, verbose = 0, batch_size = 16)
    
    
    
    # Returns the probability distribution over the possible moves
    def predict(self, state):
        
        # Setting the dimensions of the state
        x = np.array(self.state_to_arr( state))
        x = np.expand_dims(x,0)
        
        # Feed the model a int array of the state
        return self.model( x )
    
    
    
    # Returns the move that agent decides to make
    # state - current state of the board
    def policy(self, state):
        
        # Get the numpy array of distributions
        pd = np.array(self.predict(state))[0]
    
        # Not all the moves in there are possible, need to normalize the output
        # Set all pd values where state is not zero to zero (e.g. where the move is not possible)
        for oi, si in zip(range( len(pd)), state[1:]):
            if si != "0":    pd[oi] = 0    
     
        # Now, normalize the pd   
        if sum(pd) == 0:
            # A small chance that 
            pd = np.array(self.predict(state))[0]
        else:
            pd = [i/sum(pd) for i in pd ]
        
        return pd
    
    
    
    # Saves the NN information to a file
    def save(self, e):
        self.model.save('./test/m' + str(e))
    
    # Load the model
    def load(self, e):
        self.model = tf.keras.models.load_model(e)
        
    
    
    # Helpers
    
    # Takes in the board_state and turns it into the array of ints
    def state_to_arr(self, board_state):
        return [int(i) for i in tuple(board_state)]
    



            
            
#scipy.special.softmax

           
            
            
            
            
            
            
            
            