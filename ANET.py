# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:14:05 2021

@author: babay
"""

import tensorflow as tf


class ANET():
    
    # The model will be accessible directly
    model = None
    
    
    # Constructor
    # lrate - learning rate of the NN
    # layers - a list with number of nodes in the layers, alternating with activation functions. The first value is the input shape only. Activating functions accepted - "lin" - linear , "sig" - sigmoid, "tan" - tanh, "rel" - RELU
    # optimizer - the optimizer used. Following are accepted - "ADA" - Adagrad; "SGD" - Stochastic Gradient Descent; "RMS" - RMSProp; "ADAM" - Adam
    # M - after how many games the information about the network is saved into a file
    def __init__( self, lrate, layers, optimizer, M ):
        self.create_network(layers, optimizer)
    
    
    # Creates the network with specified parameters 
    # NOTE softmax activation function should be used for the output
    def create_network(self, layers, optimizer):
        
        # Create the model
        self.model = tf.keras.models.Sequential()

        # Adding the input layer
        # TODO check what shape should be here
        self.model.add(tf.keras.layers.InputLayer(input_shape = (int(layers[0]),) ) )

        # TODO use the activation functions specified in the layers argument 
        # Adding layers with number of nodes as specified in layers argument
        for i in range(len(layers[1:])):

            # Care only for the odd values
            if i % 2 != 0:
                self.model.add( tf.keras.layers.Dense( units = layers[i+1], activation = tf.nn.relu) )
    
        # Compile the model
        self.model.compile(optimizer = optimizer)
        
        
    # Train the network
    def fit(self):
        pass
    
    
    # Returns the probability distribution over the possible moves
    # the first value is for 0,0; then 0,1 ; 0,2 . etc. e.g. col/row in accessing the gridp[col][row]
    def pd(self):
        pass
    
    
    # Returns the move that agent decides to make
    # pd - probability distruvtion
    # grate - e-greedy rate 
    def policy(self, pd, grate):
        pass
    
    
    # Normalizes the output of the NN
    def normalize(self, grid, output_distribution):

        state_compact = grid.get_state(compact = True)
        
        # q - number of spots in the state that are taken. e.g. total spots minus free spots
        q = grid.size - state_compact.count('0')
        
        for output, position in zip(output_distribution, state_compact):
            pass
        
        pass
    
    
    # Saves the NN information to a file
    def save_NN(self):
        pass
    
    
    
    