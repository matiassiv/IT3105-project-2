from utils.board import Board, BoardType
from utils.hex_board_graph import HexBoardGraph
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
        self.graph = HexBoardGraph(
            self.neighbour_dict, self.board, self.board_type)
        
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
        
        self.finished = False

        # Display options
        self.graphing_freq = graphing_freq
        self.display_game = display_game

        if self.display_game:
            self.init_graph()
            self.display_graph()
    
    def generate_legal_moves(self):
        legal_moves = []
        for coord, content in self.game_state:
            if content == BoardCell.EMPTY:
                legal_moves.append(coord)
        
        return legal_moves

    def check_victory_condition(self):
        red = False
        blue = False

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
                return True

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
                return True

    def find_line(self, start, colour, prev):
        """
        Implement a fairly standard depth-first search to
        find a winning line for a player. Since only one player
        can win, we don't need to check for the shortest, or most
        optimal line, so a DFS is sufficient
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

    def make_move(self, move):

        # Update game state with selected move
        self.game_state[move] = self.turn

        # Check if game is finished
        self.finished = self.check_victory_condition()
        
        # Update turn to next player
        if self.turn == BoardCell.RED:
            self.turn = BoardCell.BLUE
        else:
            self.turn = BoardCell.RED

        # Update game display
        if self.display_game:
            # self.display_board_state()
            self.update_graph()

    def get_flattened_state(self):
        flattened = [self.turn]
        for r in range(self.size):
            for c in range(self.size):
                flattened.append(self.game_state[(r,c)])
        
        return tuple(flattened)

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

    def display_board_state(self):
        # print legal moves for debugging purposes
        for i, move in enumerate(self.legal_moves):
            print(i, move)

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
    print(test1.get_flattened_state())
    while not test1.finished:
        move = input("Select your move: ")
        if move == 'q':
            break
        r,c = int(move[0]), int(move[1])
        test1.make_move((r,c))
        print(test1.get_flattened_state())

    