from utils.board import Board, BoardType
from utils.hex_board_graph import HexBoardGraph
import networkx as nx
import matplotlib.pyplot as plt

class BoardCell:
    EMPTY = 1
    RED = 2
    BLUE = 3

class HexGame(Board):
    def __init__(
        self, 
        board_type=BoardType.DIAMOND, 
        size=6,
        turn=BoardCell.RED,  
        graphing_freq=5, 
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

        # Display options
        self.graphing_freq = graphing_freq
        self.display_game = display_game

        if self.display_game:
            self.init_graph()
            self.display_graph()
    
    def generate_legal_moves(self):
        pass

    def check_victory_condition(self):
        pass

    def make_move(self, move):

        # Update game state with selected move
        self.game_state[move] = self.turn

        # Check if game is finished
        self.check_victory_condition()
        
        # Update turn to next player
        if self.turn == BoardCell.RED:
            self.turn = BoardCell.BLUE
        else:
            self.turn = BoardCell.RED

        # Update game display
        if self.display_game:
            # self.display_board_state()
            self.update_graph()

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
    test1 = HexGame(display_game=True)
    test1.make_move((0,0))
    test1.make_move((3,4))