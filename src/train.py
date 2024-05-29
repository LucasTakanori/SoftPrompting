from arg_parser import ArgsParser
import logging
import datetime
import torch
import random
import numpy as np
from torch import optim
import os
from utils import get_memory_info



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
    def load_checkpoint(self):
        """Load trained model checkpoints to continue its training."""
        # Load checkpoint
        checkpoint_path = os.path.join(
            self.params.checkpoint_file_folder, 
            self.params.checkpoint_file_name,
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        logger.info(f"Checkpoint loaded.") 

    def load_optimizer(self):
        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )       
            
        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()
        logger.info(f"Optimizer {self.params.optimizer} loaded!")

    def load_checkpoint_params(self):
        logger.info("Loading checkpoint parameters...")
        self.params = self.checkpoint["settings"]
        logger.info("Checkpoint parameters loaded!")
 
    def set_random_seed(self):

        logger.info("Setting random seed...")

        random.seed(1234)
        np.random.seed(1234)

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        logger.info("Random seed setted.")        


    def info_mem(self, step = None, logger_level = "INFO"):

        '''Logs CPU and GPU free memory.'''
        
        cpu_available_pctg, gpu_free = get_memory_info()
        if step is not None:
            message = f"Step {self.step}: CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        else:
            message = f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        
        if logger_level == "INFO":
            logger.info(message)
        elif logger_level == "DEBUG":
            logger.debug(message)


    def initialize_training_variables(self):
        
        logger.info(f'Initializing training variables... Checkpoint: {self.params.load_checkpoint} ')
        
        if self.params.load_checkpoint:
            self.starting_epoch = self.load_checkpoint()
            self.load_checkpoint()
            # TODO: finish
            #
        else:   
            self.starting_epoch = 0
            self.step = 0 
            self.validations_without_improvement = 0 
            self.validations_without_improvement_or_opt_update = 0 
            self.early_stopping_flag = False
            self.train_loss = None
            self.training_eval_metric = 0.0
            self.validation_eval_metric = 0.0
            self.best_train_loss = np.inf
            self.best_model_train_loss = np.inf
            self.best_model_training_eval_metric = 0.0
            self.best_model_validation_eval_metric = 0.0

        self.total_batches = len(self.training_generator)
        logger.info("Training variables initialized.")

    def eval_and_save_best_model(self):

        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if founded)...')

            # Calculate the evaluation metrics
            self.evaluate()

            # Have we found a better model? (Better in validation metric).
            if self.validation_eval_metric > self.best_model_validation_eval_metric:

                logger.info('We found a better model!')

               # Update best model evaluation metrics
                self.best_model_train_loss = self.train_loss
                self.best_model_training_eval_metric = self.training_eval_metric
                self.best_model_validation_eval_metric = self.validation_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")

                self.save_model() 

                # Since we found and improvement, validations_without_improvement and validations_without_improvement_or_opt_update are reseted.
                self.validations_without_improvement = 0
                self.validations_without_improvement_or_opt_update = 0


    def train_single_epoch(self, epoch):
        logger.info(f"Starting epoch {epoch} of {self.params.max_epochs}.")

        self.net.train()

        for self.batch_number, batch_data in enumerate(self.training_generator):
            input, transcription = batch_data

            input, transcription = input.float().to(self.device), transcription.long().to(self.device)
                      
            if self.batch_number == 0: logger.info(f"input.size(): {input.size()}")

        # Calculate the prediction and the loss:
        prediction = self.net(input)
        self.loss = self.loss_function(prediction, transcription)
        self.train_loss = self.loss.item()

        # Backpropagation
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Evaluate and save the best model
        self.eval_and_save_best_model()

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