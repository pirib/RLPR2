# -*- coding: utf-8 -*-

# In-house stuff
import MCTS as mt
import ANET as an


# For general funciton
import random
import time


class RL():

    # Constructor
    def __init__(self, board_size, episodes, num_search_games, rollout_policy, grate, grate_const, c, c_const, minibatch_size, M, nn_layers, nn_optimizer, save_path):
        
        self.board_size = board_size        
        
        # 2. Replay Buffer will store the potential training cases for the ANET
        self.RBUF = {}
        
        # 3. Initialize the Neural Network (adding the input layer which is the square of the board size )
        # self.ANET = an.ANET( [board_size**2] + nn_layers + [board_size**2] , nn_optimizer, save_path)
        self.ANET = an.ANET()
        self.ANET.load(400, "6x6ABABTU")
        
        # 4. Start working through episodes/epochs
        for e in range(episodes):
            
            # Just so we are aware of the training process
            print("Episode " + str(e))
            
            # a. b. c. Initialize the MCTS
            self.mcts = mt.MCTS(    anet = self.ANET,
                                    board_size = board_size, 
                                    rollout_policy = rollout_policy,
                                    c = c,
                                    grate = grate)

            # Grate drops every episode
            if not grate_const and not grate < 0.15:
                grate *= 0.99
            
            # C for UCT drops every episode
            if not c_const and not c < 0.15:
                c *= 0.99

            print("\tPlaying...")
            # d. Run it while the board is not in terminal state
            while not self.mcts.bb.is_terminal()[0]:
                
                # Start running mcts with the root (does so by the default). Uses ANET by default. Rollouts are done num_search_games times.                 
                # Run MCTS for num_search_games times
                for sg in range(num_search_games): 
                    self.mcts.run()

                # Adding training data for the ANET to learn from 
                self.RBUF[str(self.mcts.bb.get_player()) + self.mcts.root.state ] = self.mcts.root.get_visits() 

                # Use tree policy (full greedy choice with the highest value of the action), move the board to the new state, and make the new successor state the root
                chosen_an = self.mcts.tree_policy(self.mcts.root, grate = 1)
                self.mcts.bb.make_move(chosen_an.action)
                self.mcts.root = chosen_an.child


            # Now that we are collecting a database of cool ass moves, we need to train our network with a random minibatch from there
            # e. Train ANET from the RBUF       
            print("\tTraining...")         
            for i in range(minibatch_size):
                # Pick a random training case and train the ANET
                case = random.choice( list(self.RBUF.items()) )
                self.ANET.train( case[0], case[1] )
                
            # f. Save the parameters of the NN for the evaluation
            if e % M == 0:
                print("\tSaving...")
                self.ANET.save(e, save_path)
                
                    
# Want to record how much it is going to take        
start = time.time()

# Start the training
rl = RL(
        board_size = 6, 
        episodes = 5000, # 250 
        num_search_games = 1000,
        rollout_policy = "n", 
        
        grate = 1, 
        grate_const = False,
        
        c = 1,
        c_const = False,
        
        minibatch_size = 32,

        M = 50, # 50 
        
        nn_layers = [64, "relu", 128, "relu", 256, "relu", 128, "relu", 64, "relu"], 
        nn_optimizer = "Adam",
        save_path = "D6x6Poggers"
)


# Calculate elapsed time
end = time.time()
print(end - start)






