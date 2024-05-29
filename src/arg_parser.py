import argparse
from settings import TRAIN_DEFAULT_SETTINGS

class ArgsParser:
    
    def __init__(self) -> None:
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Train a Text and Speech Emotion Recognition model.',
            )
        
    def add_parser_args(self):

        self.parser.add_argument(
            "--seed",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["random_seed"],
            help="Random seed for reproducibility",
        )
    
        self.parser.add_argument(
            '--max_epochs',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['max_epochs'],
            help = 'Max number of epochs to train.',
            )
            
        self.parser.add_argument(
            '--use_weights_and_biases',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['use_weights_and_biases'],
            help = 'Set to True if you want to use Weights and Biases.',
            )

        self.parser.add_argument(
            '--eval_and_save_best_model_every', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['eval_and_save_best_model_every'],
            help = "The model is evaluated on train and validation sets and saved every eval_and_save_best_model_every steps. \
                Set to 0 if you don't want to execute this utility.",
            )                
        self.parser.add_argument(
            '--load_checkpoint',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['load_checkpoint'],
            help = 'Set to True if you want to load a previous checkpoint and continue training from that point. \
                Loaded parameters will overwrite all inputted parameters.',
            )          

            