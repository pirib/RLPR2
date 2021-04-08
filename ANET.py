# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 13:14:05 2021

@author: babay
"""

import tensorflow as tf
import numpy as np
import random

class ANET():
    

    # Constructor
    # layers - a list with number of nodes in the layers, alternating with activation functions. The first value is the input shape only. Activating functions accepted - "lin" - linear , "sig" - sigmoid, "tan" - tanh, "rel" - RELU
    # optimizer - the optimizer used. Following are accepted - "ADA" - Adagrad; "SGD" - Stochastic Gradient Descent; "RMS" - RMSProp; "ADAM" - Adam
    # M - after how many games the information about the network is saved into a file
    def __init__( self,  layers, optimizer):
        # The model will be accessible directly
        self.model = None
        # Create the network
        self.create_network(layers, optimizer)
    
    
    # Creates the network with specified parameters 
    # NOTE softmax activation function should be used for the output
    def create_network(self, layers, optimizer):
        
        # Create the model
        self.model = tf.keras.Sequential()

        # Adding the input layer
        self.model.add(tf.keras.Input(input_shape = (layers[0],) ) )

        # TODO use the activation functions specified in the layers argument 
        # Adding layers with number of nodes as specified in layers argument
        for i in range(len(layers[1:])):

            # Care only for the odd values, even are for the actionvation function
            if i % 2 != 0:
                self.model.add( tf.keras.layers.Dense( units = layers[i+1], activation = tf.nn.relu) )
    
        # Compile the model
        self.model.compile(optimizer = optimizer)
        
        
    # Train the network
    def fit(self, x, y, epochs = 1):
        self.model.fit(x, y, epochs)
    
    
    # Returns the probability distribution over the possible moves
    # the first value is for 0,0; then 0,1 ; 0,2 . etc. e.g. col/row in accessing the gridp[col][row]
    def predict(self, x):
        return self.model.predict(x)
    
    
    # Returns the move that agent decides to make
    # state - current state of the board
    def policy(self, state):
        
        # Get the numpy array of distributions
        pd = self.model(state)        
    
        # Not all the moves in there are possible, need to normalize the output
        # Set all pd values where state is not zero to zero (e.g. where the move is not possible)
        for oi, si in zip(range(len(pd)), state):
            if si != "0":    pd[oi] = 0    
     
        # Now, normalize the pd   
        pd = pd / sum(pd)

        
        

    # Saves the NN information to a file
    def save_NN(self, e):
        self.model.save_weights('./models/m' + str(e))
    
    
    
    