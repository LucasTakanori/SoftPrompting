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
            default=1234,
            help="Random seed for reproducibility",
        )
    