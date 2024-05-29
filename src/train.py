from arg_parser import ArgsParser
import logging
import datetime
import torch



#region logging
# Logging
# -------
# Set logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion


class Trainer():
    def __init__(self, trainer_params):
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')

        self.set_params(trainer_params)
        self.set_device()

    def set_params(self, input_params):
        '''Set parameters for training.'''

        logger.info('Setting parameters...')
        self.params = input_params

        logger.info('Parameters setted.')


    def set_device(self):
        '''Set torch device.'''

        logger.info('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Running on {self.device} device.")
        
        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0
        
        logger.info("Device setted.")

    def initialize_training_variables(self):
        
        logger.info(f'Initializing training variables... Checkpoint: {self.params.load_checkpoint} ')
        
        if self.params.load_checkpoint:
            self.starting_epoch = self.load_checkpoint()
            self.load_checkpoint()
            # TODO: finish  


    def train(self, starting_epoch, max_epochs):
        logger.info(f'Starting training for {self.params.max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):
            self.train_single_epoch(self.epoch)

            logger.info(f"The evaluation metric is {self.validation_eval_metric}")


            if self.early_stopping_flag == True: 
                break

        logger.info('Training finished!')

    def main(self):
        logger.info("Training with the following parameters:")
        for key, value in self.trainer_params.items():
            logger.info(f"{key}: {value}")
        self.train(self.starting_epoch, self.params.max_epochs)
        if self.params.use_weights_and_biases: self.save_model_artifact()
        if self.params.use_weights_and_biases: self.delete_version_artifacts()               




def main():
    args_parser = ArgsParser()
    args_parser()
    trainer_params = args_parser.arguments

    trainer = Trainer(trainer_params)
    trainer.main()

if __name__=="__main__":
    main()