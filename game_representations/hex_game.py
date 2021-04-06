from game_representations.utils.board import Board, BoardType
from game_representations.utils.hex_board_graph import HexBoardGraph
import networkx as nx
import matplotlib.pyplot as plt

class BoardCell:
    EMPTY = 0
    RED = 1
    BLUE = 2

class HexGame(Board):
    def __init__(
        self, 
        board_type=BoardType.DIAMOND, 
        size=6,
        turn=BoardCell.RED,  
        graphing_freq=1, 
        display_game=False
    ):
        super().__init__(board_type, size)
        
        # Initialize all cells as empty
        self.game_state = {
            coord: BoardCell.EMPTY for row in self.board for coord in row}
        
        # Set player to begin
        self.turn = turn

        # Create list of border coords for easy victory check
        self.blue_north_west = []
        self.red_north_east = []
        self.red_south_west = []
        self.blue_south_east = []

        for i in range(self.size):
            self.blue_north_west.append((i,0))
            self.red_north_east.append((0,i))
            self.red_south_west.append((self.size-1, i))
            self.blue_south_east.append((i, self.size-1))
 
        self.game_result = 0

        # Display options
        self.graphing_freq = graphing_freq
        self.display_game = display_game

        if self.display_game:
            self.graph = HexBoardGraph(
                self.neighbour_dict, self.board, self.board_type)
            self.init_graph()
            self.display_graph()
    

    # For previous version, need to bulletproof that the versions I pass
    # the state directly work as intended
    def generate_legal_moves_(self):
        legal_moves = []
        for coord, content in self.game_state:
            if content == BoardCell.EMPTY:
                legal_moves.append(coord)
        
        return legal_moves
    def check_victory_condition(self):
        
        if self.turn == BoardCell.RED:
            red = False
            # Check if red has possible victory
            for coord in self.red_north_east:
                if self.game_state[coord] == BoardCell.RED:
                    red = True
            
            possible_reds = []
            if red:
                red = False
                for coord in self.red_south_west:
                    if self.game_state[coord] == BoardCell.RED:
                        possible_reds.append(coord)
            
            for possible in possible_reds:
                if self.find_line(possible, BoardCell.RED, []):
                    print("Red has won!")
                    return BoardCell.RED

        else:
            blue = False
            # Check if blue has possible victory
            for coord in self.blue_north_west:
                if self.game_state[coord] == BoardCell.BLUE:
                    blue = True
            
            possible_blues = []
            if blue:
                blue = False
                for coord in self.blue_south_east:
                    if self.game_state[coord] == BoardCell.BLUE:
                        possible_blues.append(coord)
            
            for possible in possible_blues:
                if self.find_line(possible, BoardCell.BLUE, []):
                    print("Blue has won!")
                    return BoardCell.BLUE
        
        return 0
    def find_line(self, start, colour, prev):
        """
        Implement a fairly standard depth-first search to
        find a winning line for a player. Since only one player
        can win, we don't need to check for the shortest, or most
        optimal line, so a DFS is sufficient, but an algorithm for
        checking disjoint sets is probably faster.
        """
        # Set node as visited
        prev.append(start)
        # Set finish line according to player colour
        finish_line = self.blue_north_west
        if colour == BoardCell.RED:
            finish_line = self.red_north_east
        
        # Check if we have reached a line to the finish
        if start in finish_line:
            return True
        
        # Check all neighbours of node
        for coord in self.neighbour_dict[start].values():
            # If neighbour is of same colour and not already visited
            # then call function recursively until we find the line
            if (self.game_state[coord] == colour \
                and coord not in prev):
                return self.find_line(coord, colour, prev)
        
        return False
    def get_game_state(self):
        flattened = [self.turn]
        for r in range(self.size):
            for c in range(self.size):
                flattened.append(self.game_state[(r,c)])
        
        return tuple(flattened)
    def update_game_state(self, move):

        # Update game state with selected move
        self.game_state[move] = self.turn

        # Check if game is finished
        self.game_result = self.check_victory_condition()
        
        if not self.game_result:
            # Update turn to next player
            if self.turn == BoardCell.RED:
                self.turn = BoardCell.BLUE
            else:
                self.turn = BoardCell.RED

        # Update game display
        if self.display_game:
            # self.display_board_state()
            self.update_graph()

    # For the new version, where states are passed through functions, rather than
    # kept as part of the object
    def generate_legal_moves(self, state):
        # Returns a one-hot encoded array over legal moves
        return [1 if c == BoardCell.EMPTY else 0 for c in state[1:]]

    def generate_next_state(self, state, action):
        new_state = list(state)

        # Update turn
        new_state[0] = new_state[0] % 2 + 1

        #Place new stone
        i = self.coord_to_state_index(action)
        new_state[i] = state[0]
        return tuple(new_state)

    def check_game_ended(self, state):
        """
        Takes in a game state of the form:
        (turn, (r0, c0), (r0, c1), ..., (r0, csize), ..., (rsize, c_size))
        and checks whether the state is a final state.
        Returns 0 for non-terminal else the player ID of the victor
        """
        # Can also add check to see whos turn it is, as only the player who
        # placed the last stone can actually win. Might do this if MCTS is slow
        
        red = False
        # Check if red has possible victory
        for coord in self.red_north_east:
            i = self.coord_to_state_index(coord)
            if state[i] == BoardCell.RED:
                red = True
        
        possible_reds = []
        
        if red:
            red = False
            for coord in self.red_south_west:
                i = self.coord_to_state_index(coord)
                if state[i] == BoardCell.RED:
                    possible_reds.append(coord) 

        if self.find_line_(state, possible_reds, BoardCell.RED):
            #print("Red has won!")
            return BoardCell.RED

        blue = False
        # Check if blue has possible victory
        for coord in self.blue_north_west:
            i = self.coord_to_state_index(coord)
            if state[i] == BoardCell.BLUE:
                blue = True
        
        possible_blues = []
        
        if blue:
            blue = False
            for coord in self.blue_south_east:
                i = self.coord_to_state_index(coord)
                if state[i] == BoardCell.BLUE:
                    possible_blues.append(coord)
        
        if self.find_line_(state, possible_blues, BoardCell.BLUE):
            #print("Blue has won!")
            return BoardCell.BLUE
        
        return 0

    def find_line_(self, state, stack, colour):
        if colour == BoardCell.RED:
            finish_line = self.red_north_east
        else:
            finish_line = self.blue_north_west

        visited = []
        while len(stack):
            v = stack.pop()
            if v not in visited:
                visited.append(v)
                for coord in self.neighbour_dict[v].values():
                    # If neighbour is of same colour then we add it to the
                    # stack. If the neighbour is on the other side, then we
                    # have found a winning line!
                    i = self.coord_to_state_index(coord)
                    if (state[i] == colour):
                        if coord in finish_line:
                            return True
                        stack.append(coord)
        return False


    '''

    def find_line_(self, state, start, colour, prev):
        """
        Implement a fairly standard depth-first search to
        find a winning line for a player. Since only one player
        can win, we don't need to check for the shortest, or most
        optimal line, so a DFS is sufficient, but an algorithm for
        checking disjoint sets is probably faster.
        """
        # Set node as visited
        prev.append(start)
        # Set finish line according to player colour
        finish_line = self.blue_north_west
        if colour == BoardCell.RED:
            finish_line = self.red_north_east
        
        if len(prev) > 5:
            print("-------------------")
            if colour == BoardCell.RED:
                print("RED:")
                print(self.red_south_west)
            else:
                print("BLUE:")
                print(self.blue_south_east)
            print(prev)
            print(finish_line)
        # Check if we have reached a line to the finish
        if start in finish_line:
            return True
        print(self.neighbour_dict[start].values())
        # Check all neighbours of node
        for coord in self.neighbour_dict[start].values():

            # If neighbour is of same colour and not already visited
            # then call function recursively until we find the line
            i = self.coord_to_state_index(coord)
            if (state[i] == colour \
                and coord not in prev):
                return self.find_line_(state, coord, colour, prev)
        
        return False
    '''
    # Some util functions to convert fram tuple representation
    # to the actual coordinates of the board and back
    def one_hot_to_action(self, index):
        return (index//self.size, index%self.size)
    def coord_to_state_index(self, coord):
        return coord[0]*self.size + coord[1] + 1
    def state_to_board(self, state):
        s = state[1:]
        for r in range(self.size):
            print(s[r*self.size:(r+1)*self.size])
    """Display methods and visualisation"""
    def init_graph(self):
        plt.clf()
        plt.ion()
        self.display_graph()
        plt.show()
        plt.pause(self.graphing_freq)

    def update_graph(self):
        plt.clf()
        self.display_graph()
        plt.pause(self.graphing_freq)

    def display_graph(self):
        nx.draw(
            self.graph.graph,
            pos=self.graph.pos,
            node_color=self.get_node_colours(),
            node_size=self.get_node_sizes()
        )

    def get_node_sizes(self):
        sizes = []
        for node in self.graph.graph:
            sizes.append(200)
        return sizes

    #TODO Change node colours to red and blue, with white
    # outline and white if nothing occupies it.
    def get_node_colours(self):
        colours = []
        for node in self.graph.graph:
            if self.game_state[node] == BoardCell.EMPTY:
                colour = "#cccccc"
            elif self.game_state[node] == BoardCell.RED:
                colour = "#db0f0f"
            elif self.game_state[node] == BoardCell.BLUE:
                colour = "#0a14d1"
            colours.append(colour)
        return colours

if __name__ == "__main__":
    test1 = HexGame(size=5, display_game=True)
    state = test1.get_game_state()
    print(test1.generate_legal_moves(state))
    while not test1.game_result:
        move = input("Select your move: ")
        if move == 'q':
            break
        r,c = int(move[0]), int(move[1])
        test1.update_game_state((r,c))
        state = test1.get_game_state()
        print(test1.generate_legal_moves(state))

    