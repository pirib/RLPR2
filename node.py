class Node:
    
    # Cosntructor
    def __init__(self, row, col, grid_size):
        # Setting parameters
        self.col = col
        self.row = row
        self.piece = 0                  # 0 - empty, 1 - white, 2 - black
        self.neighbours = []   
        
        
    # Set nodes neighbours depending on its placement and total grid_size
    def set_neighbours(self, neighbours):
        self.neighbours = neighbours
        
        
    # Puts in a pin, player 1 is white, 2 is black
    def insert_piece(self, player):
        self.piece = int(player)
    
