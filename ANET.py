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
    # layers - a list with number of nodes in the layers, alternating with activation functions. Activating functions accepted - "lin" - linear , "sig" - sigmoid, "tan" - tanh, "rel" - RELU
    # optimizer - the optimizer used. Following are accepted - "ADA" - Adagrad; "SGD" - Stochastic Gradient Descent; "RMS" - RMSProp; "ADAM" - Adam
    # M - after how many games the information about the network is saved into a file
    def __init__( self, lrate, layers, optimizer, M ):
        self.create_network(layers, optimizer)
    
    
    # Creates the network with specified parameters 
    def create_network(self, layers, optimizer):
        
        # Create the model
        self.model = tf.keras.models.Sequential()

        # Adding the input layer
        # TODO check what shape should be here
        self.model.add(tf.keras.layers.InputLayer(input_shape = (int(layers[0]),) ) )

        # TODO if adding the input later is done before, make sure to remove it from here
        # TODO use the activation functions specified in the layers argument 
        # Adding layers with number of nodes as specified in layers argument
        for i in range(len(layers)):

            # Care only for the odd values
            if i % 2 != 0:
                self.model.add( tf.keras.layers.Dense( units = layers[i], activation = tf.nn.relu) )
    
        # Compile the model
        self.model.compile(optimizer = optimizer)
        
    # Train the network
    def fit(self):
        pass
    
    # Normalizes the output of the NN
    def normalize(self, grid, output_distribution):

        state_compact = grid.get_state(compact = True)
        
        # q - number of spots in the state that are taken. e.g. total spots minus free spots
        q = grid.size - state_compact.count('0')
        
        for output, position in zip(output_distribution, state_compact):
            pass
        
        pass
    
    
    
    
    
    
    