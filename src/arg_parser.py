import argparse
from settings import TRAIN_DEFAULT_SETTINGS

class ArgsParser:
    
    def __init__(self) -> None:
        self.initialize_parser()

    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Try soft prompting with whisper',
            )
        
    def add_parser_args(self):

        self.parser.add_argument(
            "--utterances_path",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["utterances_path"],
            help="Path to the dataset",
        )
        self.parser.add_argument(
            "--validation_utterances_path",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["validation_utterances_path"],
            help="Path to the validation dataset",
        )
        
        self.parser.add_argument(
            '--checkpoint_file_folder',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["checkpoint_file_folder"],
            help='Folder to save/load model checkpoints'
        )

        self.parser.add_argument(
            '--checkpoint_file_name',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["checkpoint_file_name"],
            help='Name of the checkpoint file'
        )

        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["num_workers"],
            help="Number of workers to use for data loading",
        )

        self.parser.add_argument(
            "--loss",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["loss"],
            help="Loss function to use",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=TRAIN_DEFAULT_SETTINGS["batch_size"],
            help="Batch size for training",
        )

        self.parser.add_argument(
            "--random_crop_secs",
            type=float,
            default=TRAIN_DEFAULT_SETTINGS["random_crop_secs"],
            help="Random crop seconds for training",
        )

        self.parser.add_argument(
            "--padding_type",
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["padding_type"],
            help="Padding type for cropping",
        )

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
            '--asr_model',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS['asr_model'],
            help='The ASR model to use.',
        )

        self.parser.add_argument(
            '--tokens_max_length',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS['tokens_max_length'],
            help = 'Maximum length of the tokens in the Whisper decoder.',
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

        self.parser.add_argument(
            '--training_augmentation_prob', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['training_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )

        self.parser.add_argument(
            '--evaluation_augmentation_prob', 
            type = float, 
            default = TRAIN_DEFAULT_SETTINGS['evaluation_augmentation_prob'],
            help = 'Probability of applying data augmentation to each file. Set to 0 if not augmentation is desired.'
            )
        self.parser.add_argument(
            '--sample_rate', 
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS['sample_rate'],
            help = "Sample rate that you want to use (every audio loaded is resampled to this frequency)."
            ) 

        self.parser.add_argument(
            '--whisper_flavour',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS["whisper_flavour"],
            help = 'The whisper flavour to use.'
            )         
        
        self.parser.add_argument(
            '--prompt_use_rate',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS["prompt_use_rate"],
            help = 'The rate at which prompts are used during training.'
        )

        self.parser.add_argument(
            '--prompt_length',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS["prompt_length"],
            help = 'The length of the prompt.'
        )
        
        self.parser.add_argument(
            '--speech_representation',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS["speech_representation"],
            help = 'The type of speech representation to use (e.g., "mel" for mel spectrogram).'
        )
        self.parser.add_argument(
            '--nmels',
            type = int, 
            default = TRAIN_DEFAULT_SETTINGS["nmels"],
            help = 'The number of mels windows used for creating mel-spectrogram'
        )

        self.parser.add_argument(
            '--context_len',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS["context_len"],
            help = 'The context len of the tokens. In the code it is represented also as the context len of the softprompts'
        )

        self.parser.add_argument(
            '--optimizer',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS["optimizer"],
            help = 'What optimizer is used to train the model'
        )
        self.parser.add_argument(
            '--learning_rate',
            type = float,
            default = TRAIN_DEFAULT_SETTINGS["learning_rate"],
            help = 'Learning rate of the optimizer'
        )
        self.parser.add_argument(
            '--soft_prompt_location',
            type=str,
            default=TRAIN_DEFAULT_SETTINGS["encoder"],
            choices=["encoder", "decoder", "both"],
            help='Where to apply soft prompts: "encoder" or "decoder" or "both"'
        )
        self.parser.add_argument(
            '--vocab_size',
            type = int,
            default = TRAIN_DEFAULT_SETTINGS["vocab_size"],
            help = "The vocabulary size of the ASR text decoder."
        )
        self.parser.add_argument(
            '--use_weights_and_biases',
            action = argparse.BooleanOptionalAction,
            default = TRAIN_DEFAULT_SETTINGS['use_weights_and_biases'],
            help = 'Set to True if you want to use Weights and Biases.',
            )

        self.parser.add_argument(
            '--wandb_project',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['wandb_project'],
            help = 'The name of the Weights and Biases project.',
            )

        self.parser.add_argument(
            '--wandb_entity',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['wandb_entity'],
            help = 'The Weights and Biases username or team name.',
            )

        self.parser.add_argument(
            '--wandb_run_name',
            type = str,
            default = TRAIN_DEFAULT_SETTINGS['wandb_run_name'],
            help = 'The name of the Weights and Biases run.',
            )



    def main(self):
        self.add_parser_args()
        self.arguments = self.parser.parse_args()
    
    def __call__(self):
        self.main()