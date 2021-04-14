import matplotlib.pyplot as plt
from torch import nn
import torch
import numpy as np
import random
from state_manager import StateManager
from mcts import MCTS
from ann import ANN
import config as cfg
"""
Duplicate installs of some library on old computer, so need to set this
flag to use it for training
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Function for the entire RL with ANN and MCTS
"""


def loss_p(outputs, targets):
    return -torch.sum(targets * torch.log(1e-9 + outputs)) / targets.size()[0]

class Trainer:
    def __init__(self, 
        game_params=cfg.TRAIN_game_params,
        num_episodes=100,
        num_models_saved=5,
        save_path="trained_models/temporary/",
        display_first=True,
        mcts_params=cfg.TRAIN_mcts,
        ann_params=cfg.DEMO_nn
        ):

        self.game_params = game_params
        self.num_episodes = num_episodes
        self.save_interval = num_episodes // (num_models_saved-1)
        self.save_path = save_path
        self.display_game = game_params["display_game"]
        self.display_first = display_first
        self.mcts_params = mcts_params
        self.ann_params = ann_params
        
        game = StateManager(game_params=game_params)
        self.size = game.get_game_size()
        input_size = self.size
        output_size = len(game.generate_legal_moves(game.get_game_state()))
        self.ann = ANN(input_size, output_size, ann_type=ann_params["ann"])
        print(self.ann.model)

        # Set loss function and optimizer for training session
        self.loss_fn = loss_p
        self.set_optimizer()
    
        self.replay_buffer = []
        self.accs = []
        self.losses = []
        self.results = {}
        self.results[1] = [0, 0]
        self.results[2] = [0, 0]


    def reinforcement_learner(self):

        turn = 1
        # TODO find way of passing game params
        game = StateManager(turn, game_params=self.game_params, display_game=self.display_first)
        m = MCTS(
            game, 
            self.ann, 
            search_time=self.mcts_params["search_time"], 
            eps=self.mcts_params["rollout_exploration"],
            c_ucb=self.mcts_params["c_ucb"]
            )
        i = 0
        s = game.get_game_state()

        # Iterate over number of episodes
        while i <= self.num_episodes:
            self.ann.model.eval()
            if len(self.losses) > 0:
                print(i, len(self.replay_buffer), self.losses[-1], self.accs[-1], flush=True)
            else:
                print(i, len(self.replay_buffer), flush=True)
            if i % self.save_interval == 0:
                torch.save(self.ann.model.state_dict(),
                        self.save_path+"iteration_"+str(i)+".pt")
          
            while True:
                action_prob = m.getActionProb(s)
                self.replay_buffer.append(
                    (self.ann.convert_state_to_input(s, self.size), tuple(action_prob)))
                
                # Become more exploitative throughout the training
                r = np.random.rand()
                if r > i/self.num_episodes:
                    action = np.random.choice(
                        len(action_prob), p=action_prob)
                else:
                    action = np.argmax(action_prob)
                
                a = game.one_hot_to_action(action)
                s = game.generate_next_state(s, a)
                if self.display_first:
                    game.update_game_state(a)
                result = game.check_game_ended(s)
                if result:
                    self.results[turn][result-1] += 1
                    if self.display_first:
                        print(f"Player {result} won the first game!")
                    break
            
            self.display_first = False
            if self.game_params["display_game"]:
                # If all games are to be displayed
                self.display_first = True
            # Get random minibatch and train model 1 step
            if len(self.replay_buffer) > 64:
    
                self.ann.model.train()     
                batch = random.sample(self.replay_buffer, 64)
                loss, acc = self.ann.train_step(self.loss_fn, self.optimizer, batch)
                
                self.losses.append(loss)
                self.accs.append(acc)

                i += 1
                
                if len(self.replay_buffer) > 2000:
                    # Remove early games from buffer using a sliding window of 2000 states
                    self.replay_buffer = self.replay_buffer[64:]
                
            turn = turn % 2 + 1
            game = StateManager(turn, self.game_params, display_game=self.display_first)
            s = game.get_game_state()
            m = MCTS(
                game, 
                self.ann, 
                search_time=self.mcts_params["search_time"], 
                eps=self.mcts_params["rollout_exploration"],
                c_ucb=self.mcts_params["c_ucb"]
                )

    def plot_training_results(self):
        print(self.results)
        x = np.arange(len(self.losses))
        plt.close()
        plt.plot(x, self.losses)
        plt.savefig(self.save_path+"losses.png")
        plt.clf()
        plt.plot(x, self.accs)
        plt.savefig(self.save_path+"accs.png")
        plt.close()
    
    def set_optimizer(self):
        if self.ann_params["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.ann.model.parameters(), lr=self.ann_params["lr"]
            )
        elif self.ann_params    ["optimizer"] == "adagrad":
            self.optimizer = torch.optim.Adagrad(
                self.ann.model.parameters(), lr=self.ann_params["lr"]
            )
        elif self.ann_params["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.ann.model.parameters(), lr=self.ann_params["lr"]
            )
        elif self.ann_params["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.ann.model.parameters(), lr=self.ann_params["lr"]
            )
        

if __name__ == "__main__":
    trainer = Trainer(
        game_params=cfg.TRAIN_game_params,
        num_episodes=cfg.TRAINER_SETTINGS["num_episodes"],
        num_models_saved=cfg.TRAINER_SETTINGS["num_models_saved"],
        save_path=cfg.TRAINER_SETTINGS["save_path"],
        display_first=cfg.TRAINER_SETTINGS["display_first"],
        mcts_params=cfg.TRAIN_mcts,
        ann_params=cfg.DEMO_nn
    )
    trainer.reinforcement_learner()
    trainer.plot_training_results()
