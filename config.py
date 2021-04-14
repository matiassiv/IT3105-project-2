nim_settings = {
    "N": 71,
    "K": 4
}
#TRAINING SETTINGS
TRAIN_game_params = {
    "game_type": "hex",     # Set which game to play
    "size": 5,              # Set size of game
    "graphing_freq": 0.5,     # Set update frequency of graph
    "turn": 1,              # Set turn of starting player
    "display_game": False,  # Set to turn on game display continually, RECOMMEND FALSE
}

TRAIN_mcts = {
    "search_time": 0.4,
    "rollout_exploration": 0.3,
    "c_ucb": 1.2
}

TRAINER_SETTINGS = {
    "num_episodes": 20,
    "num_models_saved": 5,
    "save_path": "trained_models/demo/",
    "display_first": True
}

# NN Demo with adjustable layers
DEMO_nn = {
    "ann": "hex_demo",                             # nn architecture - leave as hex_demo
    "optimizer": "rmsprop",                        # adam, sgd, adagrad, rmsprop
    "lr": 5e-3,
    # Number of nodes in hidden layers - Must match number of activation functions
    "hidden": [30, 20],                        
    "activation_funcs": ["relu", "tanh"],    # relu, sigmoid, linear, tanh
}

TOPP_mcts = {
    "search_time": 0.3,
    "rollout_exploration": 0.2,
    "c_ucb": 0.8
}
TOPP_SETTINGS = {
    # REMEMBER TO CHECK BOARD SIZE BEFORE STARTING TOURNAMENT
    "ann": "hex_demo",   # Select nn architecture. hex_5 for pretrained
    "num_games": 16,

    # Path to saved models. Path to pretrained models for hex 5: "trained_models/hex_5/iteration_"
    "model_path": "trained_models/demo/iteration_",

    # The contenders in the tournament. Hex_5 pretrained:   ["0.pt", "50.pt", "150.pt", "350.pt"],
    "models": ["0.pt", "5.pt", "10.pt", "15.pt", "20.pt"], #["0.pt", "5.pt", "10.pt", "15.pt", "20.pt"],

     # Whether to perform search during tournament. 
     # False tests net without MCTS and is therefore much faster
    "search": True,
    "display_first": True   # Whether we should display the first tournament game

}
