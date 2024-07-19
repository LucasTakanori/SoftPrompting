from arg_parser import ArgsParser
import logging
import datetime
import torch
import random
from typing import List
import re
import jiwer
import numpy as np
from torch import optim
import os
from utils import get_memory_info
from torch import nn
from model import PromptASR
import wandb
from data import TrainDataset
from torch.utils.data import DataLoader
from model import PromptASR
from whisper.tokenizer import get_tokenizer


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
        self.load_training_data()
        self.load_validation_data() 
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.initialize_training_variables()
        self.tokenizer = get_tokenizer(self.params.whisper_flavour)

    def set_params(self, input_params):
        '''Set parameters for training.'''

        logger.info('Setting parameters...')
        self.params = input_params

        # convert the argparse.Namespace() into a dictionary
        params_dict = vars(self.params)
        # we print the dictionary in a sorted way:
        for key, value in sorted(params_dict.items()):
            print(f"{key}: {value}")        

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
    


    def calculate_wer(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate the average Word Error Rate (WER) across all prediction-ground truth tuples,
        applying preprocessing to each string.

        Parameters:
        - predictions: A list of predicted strings.
        - ground_truths: A list of ground truth strings corresponding to the predictions.

        Returns:
        - The average WER across all tuples.
        """

        # Define preprocessing transformations
        transforms = jiwer.Compose(
            [
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ]
        )

        total_wer = 0.0
        num_pairs = len(predictions)

        # Ensure predictions and ground_truths are of equal length
        assert len(predictions) == len(ground_truths), "Predictions and ground truths lists must be of equal length."

        # Iterate over each prediction-ground truth pair
        for prediction, ground_truth in zip(predictions, ground_truths):
            # Calculate WER for the current pair after applying transformations
            wer = jiwer.wer(
                        ground_truth,
                        prediction,
                        truth_transform=transforms,
                        hypothesis_transform=transforms,
                    )
            total_wer += wer

        # Calculate average WER
        average_wer = total_wer / num_pairs if num_pairs > 0 else 0.0

        return average_wer



    # Function to calculate CER
    def calculate_cer(self, references, hypotheses):
        return jiwer.cer(references, hypotheses)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(
            self.params.checkpoint_file_folder, 
            self.params.checkpoint_file_name,
        )

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_model_train_loss = checkpoint['loss']
        self.best_model_training_eval_metric = checkpoint['training_metric']
        self.best_model_validation_eval_metric = checkpoint['validation_metric']

        logger.info(f"Checkpoint loaded. Resuming from epoch {self.epoch}")
        
        return self.epoch + 1
    

    def load_optimizer(self):
        logger.info(f"Loading the optimizer... {self.params.optimizer}") 
        #logger.info(f"self.net.parameters() type: {type(self.net.parameters)}")
        #logger.info(f"self.params type: {type(self.params)}")
        logger.info(self.net.parameters())

        #for param in filter(lambda p: p.requires_grad, self.net.parameters()):
        #    print(param.size())


        if self.params.optimizer == 'adam':
            # HACK Mirar be quins parametres ficar 

            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                #self.net.parameters.parse_args(),
                lr=self.params.learning_rate,
                #weight_decay=self.params.weight_decay
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
        #logger.info(type(self.optimizer))
        logger.info(f"Optimizer {self.params.optimizer} loaded!")

    def load_checkpoint_params(self):
        logger.info("Loading checkpoint parameters...")
        self.params = self.checkpoint["settings"]
        logger.info("Checkpoint parameters loaded!")


 
    def load_loss_function(self):
        logger.info("Loading the loss function...")

        if self.params.loss == "CrossEntropy":
            # The nn.CrossEntropyLoss() criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class
            logger.info("Using CrossEntropyLoss() as loss function.")
            self.loss_function = nn.CrossEntropyLoss()    
    
    def set_random_seed(self):

        logger.info("Setting random seed...")

        random.seed(1234)
        np.random.seed(1234)

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        logger.info("Random seed setted.")        

    def set_log_file_handler(self):

        '''Set a logging file handler.'''

        if not os.path.exists(self.params.log_file_folder):
            os.makedirs(self.params.log_file_folder)
        
        if self.params.use_weights_and_biases:
            logger_file_name = f"{self.start_datetime}_{wandb.run.id}_{wandb.run.name}.log"
        else:
            logger_file_name = f"{self.start_datetime}.log"
        logger_file_name = logger_file_name.replace(':', '_').replace(' ', '_').replace('-', '_')

        logger_file_path = os.path.join(self.params.log_file_folder, logger_file_name)
        logger_file_handler = logging.FileHandler(logger_file_path, mode = 'w')
        logger_file_handler.setLevel(logging.INFO) # TODO set the file handler level as a input param
        logger_file_handler.setFormatter(logger_formatter)

        logger.addHandler(logger_file_handler)


    def load_training_data(self):
        '''Load training data.'''
        logger.info("Loading training data...")
        training_dataset = TrainDataset(utterances_paths=self.params.utterances_path,
                                        random_crop_secs=self.params.random_crop_secs,
                                        tokens_max_length=self.params.tokens_max_length,
                                        speech_representation=self.params.speech_representation,
                                        vocab_size=self.params.vocab_size,
                                        nmels=self.params.nmels,
                                        prompt_use_rate= 0.5, #HACK learn wtf is this
                                        max_prompt_length=223, #HACK I have other metric for this. Unify
                                        context_len=self.params.context_len,
                                        augmentation_prob=self.params.training_augmentation_prob,
                                        padding_type=self.params.padding_type,
                                        whisper_flavour=self.params.whisper_flavour,
                                        sample_rate=self.params.sample_rate,
                                        waveforms_mean=None,
                                        waveforms_std=None)
        
        data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": True,
            "num_workers": self.params.num_workers,
        }

        self.training_generator = DataLoader(
            training_dataset,
            **data_loader_parameters,
        )
        del training_dataset

    def load_validation_data(self):
        '''Load validation data.'''
        logger.info('Loading validation data...')
        validation_dataset = TrainDataset(utterances_paths=self.params.validation_utterances_path,
                                               random_crop_secs=self.params.random_crop_secs,
                                               tokens_max_length=self.params.tokens_max_length,
                                               speech_representation=self.params.speech_representation,
                                               vocab_size=self.params.vocab_size,
                                               nmels=self.params.nmels,
                                               prompt_use_rate= 0.5,  # Adjust as needed
                                               max_prompt_length=223,  # Adjust as needed
                                               context_len=self.params.context_len,
                                               augmentation_prob=self.params.training_augmentation_prob,
                                               padding_type=self.params.padding_type,
                                               whisper_flavour=self.params.whisper_flavour,
                                               sample_rate=self.params.sample_rate,
                                               waveforms_mean=None,
                                               waveforms_std=None)


        eval_data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": False,
            "num_workers": self.params.num_workers,
        }

        self.eval_generator = DataLoader(
            validation_dataset,
            **eval_data_loader_parameters,
        )
        del validation_dataset

    


    def load_network(self):
        """Load the network."""
        logger.info("Loading network...")

        # TODO: load the model
        # Load model class
        #self.net = Classifier(self.params, self.device)
        
        # HACK: naive model
        self.net = PromptASR(self.params, self.device)
        logger.info(f"params in load_network {self.net.parameters}")

        if self.params.load_checkpoint == True:
            # TODO: load the model from the checkpoint if wanted
            self.load_checkpoint_network()
        
        # TODO: Fix this 
        """
        # Data Parallelism
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.net = nn.DataParallel(self.net)
        """

        self.net.to(self.device)
        
        logger.info(self.net)

        # Print the number of trainable parameters
        self.total_trainable_params = 0
        parms_dict = {}
        logger.info(f"Detail of every trainable layer:")

        for name, parameter in self.net.named_parameters():

            layer_name = name.split(".")[1]
            if layer_name not in parms_dict.keys():
                parms_dict[layer_name] = 0

            logger.debug(f"name: {name}, layer_name: {layer_name}")

            if not parameter.requires_grad:
                continue
            trainable_params = parameter.numel()

            logger.info(f"{name} is trainable with {parameter.numel()} parameters")
            
            parms_dict[layer_name] = parms_dict[layer_name] + trainable_params
            
            self.total_trainable_params += trainable_params
        
        # Check if this is correct
        logger.info(f"Total trainable parameters per layer:{self.total_trainable_params}")
        for layer_name in parms_dict.keys():
            logger.info(f"{layer_name}: {parms_dict[layer_name]}")

        #summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        logger.info(f"Network loaded, total_trainable_params: {self.total_trainable_params}")

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

            logger.info('Evaluating and saving the new best model (if found)...')

            self.evaluate()

            if self.validation_eval_metric > self.best_model_validation_eval_metric:
                logger.info('We found a better model!')

                self.best_model_train_loss = self.train_loss
                self.best_model_training_eval_metric = self.training_eval_metric
                self.best_model_validation_eval_metric = self.validation_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}")
                logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")

                self.save_model()

                self.validations_without_improvement = 0
                self.validations_without_improvement_or_opt_update = 0
            else:
                self.validations_without_improvement += 1
                self.validations_without_improvement_or_opt_update += 1

                #if self.validations_without_improvement >= self.params.early_stopping_patience:
                #    logger.info("Early stopping triggered.")
                #    self.early_stopping_flag = True

    def evaluate(self):
        self.evaluate_training()
        self.evaluate_validation()



    def logits_to_words(self, logits):
        pattern = r'<\|.*?\|>'

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Select the token with the highest probability for each position
        predicted_tokens = torch.argmax(probabilities, dim=-1)
        
        # Convert tokens to words
        predicted_words = []
        for tokens in predicted_tokens:
            words = self.tokenizer.decode(tokens.tolist())
            words = re.sub(pattern, '', words.replace('!',''))
            predicted_words.append(words)
    
        return predicted_words

    def evaluate_training(self):
        logger.info("Evaluating training task...")

        with torch.no_grad():
            self.net.eval()

            all_predictions = []
            all_ground_truths = []

            for batch_data in self.training_generator:
                input, transcription, decoder_input, ground_truth = batch_data
                input = input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                ground_truth = ground_truth.to(self.device)

                # Calculate the prediction
                prediction = self.net(input, decoder_input)

                # Convert logits to words
                text_predictions = self.logits_to_words(prediction)
                #print("prediction")
                #print(text_predictions)
                text_ground_truths = self.logits_to_words(ground_truth)
                #print("ground_truth")
                #print(text_ground_truths)
                all_predictions.extend(text_predictions)
                all_ground_truths.extend(text_ground_truths)


            # Calculate WER for all samples at once
            metric_score = self.calculate_wer(all_predictions, all_ground_truths)
            self.training_eval_metric = metric_score
            logger.info(f"Training WER: {metric_score:.3f}")

        self.net.train()

        logger.info("Training task evaluated.")
        logger.info(f"WER on training set: {self.training_eval_metric:.3f}")

    def evaluate_validation(self):
        logger.info("Evaluating validation task...")

        with torch.no_grad():
            self.net.eval()

            all_predictions = []
            all_ground_truths = []

            for batch_data in self.eval_generator:
                input, transcription, decoder_input, ground_truth = batch_data
                input = input.to(self.device)
                decoder_input = decoder_input.to(self.device)
                ground_truth = ground_truth.to(self.device)

                prediction = self.net(input, decoder_input)

                text_predictions = self.logits_to_words(prediction)
                text_ground_truths = self.logits_to_words(ground_truth)

                all_predictions.extend(text_predictions)
                all_ground_truths.extend(text_ground_truths)

            metric_score = self.calculate_wer(all_predictions, all_ground_truths)
            self.validation_eval_metric = metric_score

            logger.info(f"Validation WER: {metric_score:.3f}")

        self.net.train()

        logger.info("Validation task evaluated.")
        logger.info(f"WER on validation set: {self.validation_eval_metric:.3f}")

    def train_single_epoch(self, epoch):
        logger.info(f"Starting epoch {epoch} of {self.params.max_epochs}.")

        self.net.train()


        for self.batch_number, batch_data in enumerate(self.training_generator):
            input, transcription, decoder_input, ground_truth = batch_data
            input, transcription, decoder_input, ground_truth = input.to(self.device), transcription.to(self.device), decoder_input.to(self.device), ground_truth.to(self.device)
                      
            #if self.batch_number == 0: logger.info(f"input.size(): {input.size()}")

            # Calculate the prediction and the loss:
            prediction = self.net(input, decoder_input)

            prediction = prediction.to(self.device)
            
            #print(prediction)
            #print("prediction")
            #print(self.logits_to_words(prediction))
            #print("ground truth")
            #print(self.logits_to_words(ground_truth))
            #logger.info(f"In File train.py and function train_single_epoch() : input.size(): {input.size()}, transcription.size(): {transcription.size()}, len(prediction): {len(prediction)}") # WATCH OUT IF BATCH NUMBER >1 ERROR
            # Print predictions (add this line)
            # HACK prediction goes torch.Size([16, 448, 51865] instead of 444 just take the tensor and crop it
            #if(prediction.size(2)!=2):  prediction = prediction[:, :, :444]

            #logger.info(f"Prediction type: {prediction.dtype}")
            #logger.info(f"Ground truth type: {ground_truth.dtype}")

            #logger.info(f"Prediction size: {prediction.size()}")
            #logger.info(f"Ground truth size: {ground_truth.size()}")

            self.loss = self.loss_function(prediction, ground_truth)
            logger.info(f"loss: {self.loss}")

            self.train_loss = self.loss.item()

            # Backpropagation
            #logger.info(type(self.optimizer))
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            self.step = self.step + 1
            # Evaluate and save the best model
            self.eval_and_save_best_model()

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

    # def train(self, starting_epoch, max_epochs):
    #     logger.info(f'Starting training for {self.params.max_epochs} epochs.')

    #     try:
    #         for self.epoch in range(starting_epoch, max_epochs):
    #             self.train_single_epoch(self.epoch)

    #             logger.info(f"The evaluation metric is {self.validation_eval_metric}")

    #             if self.early_stopping_flag:
    #                 logger.info("Early stopping triggered.")
    #                 break

    #     except Exception as e:
    #         logger.error(f"An error occurred during training: {str(e)}")
    #         # Optionally, you can add code here to save the model state before exiting

    #    logger.info('Training finished!')

    def train(self, starting_epoch, max_epochs):
        logger.info(f'Starting training for {self.params.max_epochs} epochs.')


        for self.epoch in range(starting_epoch, max_epochs):
            self.train_single_epoch(self.epoch)

            logger.info(f"The evaluation metric is {self.validation_eval_metric}")

            if self.early_stopping_flag:
                logger.info("Early stopping triggered.")
                break

    def save_model(self):
        logger.info("Saving the best model...")
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_model_train_loss,
            'training_metric': self.best_model_training_eval_metric,
            'validation_metric': self.best_model_validation_eval_metric,
        }
        
        save_path = os.path.join(self.params.checkpoint_file_folder, f'best_model_epoch_{self.epoch}.pth')
        torch.save(checkpoint, save_path)
        
        logger.info(f"Best model saved to {save_path}")

    def main(self):
        logger.info("Training with the following parameters:")
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
