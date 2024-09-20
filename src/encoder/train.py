from arg_parser import ArgsParser
import logging
import datetime
import torch
import wandb
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
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

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
        self.load_network() 
        self.load_training_data()
        self.load_validation_data() 
        self.load_loss_function()
        self.load_optimizer()
        self.load_scheduler()
        self.initialize_training_variables()
        self.tokenizer = get_tokenizer(self.params.whisper_flavour)
        if self.params.use_weights_and_biases:
            wandb.init(
                project=self.params.wandb_project,
                entity=self.params.wandb_entity,
                name=self.params.wandb_run_name,
                config=vars(self.params),
                mode=self.params.wandb_mode
            )

    def set_params(self, input_params):
        '''Set parameters for training.'''

        logger.info('Setting parameters...')
        self.params = input_params

        # convert the argparse.Namespace() into a dictionary
        params_dict = vars(self.params)
        # we print the dictionary in a sorted way:
        for key, value in sorted(params_dict.items()):
            print(f"{key}: {value}")        

        logger.info('Parameters set')

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
        
        logger.info("Device set")
    
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # New line
        self.epoch = checkpoint['epoch']
        self.best_model_train_loss = checkpoint['loss']
        self.best_model_training_eval_metric = checkpoint['training_metric']
        self.best_model_validation_eval_metric = checkpoint['validation_metric']

        logger.info(f"Checkpoint loaded. Resuming from epoch {self.epoch}")
        
        return self.epoch + 1
    
    def load_optimizer(self):
        logger.info(f"Loading the optimizer... {self.params.optimizer}")
        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.net.trainable_params,
                lr=self.params.learning_rate,
            )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.net.trainable_params,  # Changed from filter(lambda p: p.requires_grad, self.net.parameters())
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
            )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.net.trainable_params,  # Changed from filter(lambda p: p.requires_grad, self.net.parameters())
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
            )
                    
        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()
        #logger.info(type(self.optimizer))
        logger.info(f"Optimizer {self.params.optimizer} loaded!")

    def load_scheduler(self):
        logger.info("Loading the learning rate scheduler...")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True
        )
        logger.info("Learning rate scheduler loaded!")

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
        training_dataset = TrainDataset(
            utterances_paths=self.params.utterances_path,
            processor=self.net.processor,
            random_crop_secs=self.params.random_crop_secs,
            tokens_max_length=self.params.tokens_max_length,
            speech_representation=self.params.speech_representation,
            prompt_use_rate=0.5, 
            prompt_length=self.params.prompt_length,  
            vocab_size=self.params.vocab_size,
            nmels=self.params.nmels,
            padding_type=self.params.padding_type,
            augmentation_prob=self.params.training_augmentation_prob,
            sample_rate=self.params.sample_rate
        )
        
        data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": True,
            "num_workers": self.params.num_workers,
        }

        self.training_generator = DataLoader(
            training_dataset,
            **data_loader_parameters
        )
        del training_dataset

    def load_validation_data(self):
        '''Load validation data.'''
        logger.info('Loading validation data...')
        validation_dataset = TrainDataset(
            utterances_paths=self.params.validation_utterances_path,
            processor=self.net.processor,
            random_crop_secs=self.params.random_crop_secs,
            tokens_max_length=self.params.tokens_max_length,
            speech_representation=self.params.speech_representation,
            prompt_use_rate=0.5,  
            prompt_length=223, 
            vocab_size=self.params.vocab_size,
            nmels=self.params.nmels,
            padding_type=self.params.padding_type,
            sample_rate=self.params.sample_rate
        )

        eval_data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": False,  # Usually, we don't shuffle the validation data
            "num_workers": self.params.num_workers,
        }

        self.eval_generator = DataLoader(
        validation_dataset,
        **eval_data_loader_parameters
        )
        logger.info('Validation data loaded.')
        del validation_dataset
    
    def load_network(self):
        logger.info("Loading network...")
        self.net = PromptASR(self.params, self.device)
        self.net.to(self.device)
        logger.info("Network loaded.")

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
            self.best_model_training_eval_metric = np.inf
            self.best_model_validation_eval_metric = np.inf

        self.total_batches = len(self.training_generator)
        logger.info("Training variables initialized.")

    def eval_and_save_best_model(self):
        if self.step > 0 and self.params.eval_and_save_best_model_every > 0 \
            and self.step % self.params.eval_and_save_best_model_every == 0:

            logger.info('Evaluating and saving the new best model (if found)...')

            self.evaluate()

            self.scheduler.step(self.validation_eval_metric)
            
            if self.validation_eval_metric < self.best_model_validation_eval_metric:  # Changed to < because lower WER is better
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

    def evaluate(self):
        self.evaluate_training()
        self.evaluate_validation()

    def logits_to_words(self, logits):
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcriptions
   
    def evaluate_training(self):
        logger.info("Evaluating training task...")
        self.net.eval()

        all_predictions = []
        all_ground_truths = []
        total_loss = 0

        with torch.no_grad():
            progress_bar = tqdm(self.training_generator, desc="Evaluating Training")
            for batch_data in progress_bar:
                input_features, labels = batch_data
                input_features = input_features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(input_features)
                
                if "generated_ids" in outputs:
                    predicted_ids = outputs["generated_ids"]
                else:
                    predicted_ids = outputs.logits.argmax(dim=-1)

                predictions = self.net.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                ground_truths = self.net.processor.batch_decode(labels, skip_special_tokens=True)

                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)

        # Calculate WER
        metric_score = self.calculate_wer(all_predictions, all_ground_truths)
        self.training_eval_metric = metric_score
        logger.info(f"Training WER: {metric_score:.3f}")

        if self.params.use_weights_and_biases:
            wandb.log({
                "train_wer": self.training_eval_metric,
            })

        self.net.train()
        logger.info(f"WER on training set: {self.training_eval_metric:.3f}")

    def evaluate_validation(self):
        logger.info("Evaluating validation task...")
        self.net.eval()

        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            progress_bar = tqdm(self.eval_generator, desc="Evaluating Validation")
            for batch_data in progress_bar:
                input_features, labels = batch_data
                input_features = input_features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(input_features)
                
                if isinstance(outputs, dict):
                    if "generated_ids" in outputs:
                        predicted_ids = outputs["generated_ids"]
                    elif "logits" in outputs:
                        predicted_ids = outputs["logits"].argmax(dim=-1)
                    else:
                        raise ValueError(f"Unexpected output format: {outputs.keys()}")
                else:
                    predicted_ids = outputs.argmax(dim=-1)

                predictions = self.net.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                ground_truths = self.net.processor.batch_decode(labels, skip_special_tokens=True)

                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)

        # Calculate WER
        metric_score = self.calculate_wer(all_predictions, all_ground_truths)
        self.validation_eval_metric = metric_score
        logger.info(f"Validation WER: {metric_score:.3f}")

        # Log some examples
        num_examples = min(5, len(all_predictions))
        for i in range(num_examples):
            logger.info(f"Example {i+1}:")
            logger.info(f"  Prediction: {all_predictions[i]}")
            logger.info(f"  Ground Truth: {all_ground_truths[i]}")

        if self.params.use_weights_and_biases:
            # Log val metric
            wandb.log({
                "val_wer": self.validation_eval_metric,
            })
            # Log some examples
            num_examples = min(5, len(all_predictions))
            example_table = wandb.Table(columns=["Prediction", "Ground Truth"])
            for pred, truth in zip(all_predictions[:num_examples], all_ground_truths[:num_examples]):
                example_table.add_data(pred, truth)
            wandb.log({"baseline_examples": example_table})

        self.net.train()
        logger.info(f"WER on validation set: {self.validation_eval_metric:.3f}")
    
    def train_single_epoch(self, epoch):
        logger.info(f"Starting epoch {epoch} of {self.params.max_epochs}.")
        self.net.train()
        total_loss = 0

        for self.batch_number, batch_data in enumerate(tqdm(self.training_generator, desc=f"Epoch {epoch}")):
            input_features, labels = batch_data
            input_features = input_features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(input_features, labels)
            loss = outputs.loss

            self.train_loss = loss.item()
            total_loss += self.train_loss

            if self.params.use_weights_and_biases:
                wandb.log({
                    "train_loss": self.train_loss,
                    "epoch": self.epoch,
                    "step": self.step,
                    "learning_rate": self.optimizer.param_groups[0]['lr']  
                })

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.trainable_params, max_norm=1.0)
            self.optimizer.step()

            self.step += 1
            if self.step % 100 == 0:
                logger.info(f"Step {self.step}, Loss: {self.train_loss:.4f}")

            if self.step % self.params.eval_and_save_best_model_every == 0:
                self.eval_and_save_best_model()

        avg_loss = total_loss / len(self.training_generator)
        logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    def train(self, starting_epoch, max_epochs):
        logger.info(f'Starting training for {max_epochs} epochs.')

        for self.epoch in range(starting_epoch, max_epochs):
            self.train_single_epoch(self.epoch)

            logger.info(f"The evaluation metric is {self.validation_eval_metric}")

            if self.early_stopping_flag:
                logger.info("Early stopping triggered.")
                break

        logger.info("Training completed.")

    def save_model(self):
        logger.info("Saving the best model...")
        
        save_dir = os.path.join(self.params.checkpoint_file_folder, self.params.wandb_run_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'best_model_epoch_{self.epoch}.pth')
        
        try:
            
            self.net.save_pretrained(save_path)
            
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),  
                'loss': self.best_model_train_loss,
                'training_metric': self.best_model_training_eval_metric,
                'validation_metric': self.best_model_validation_eval_metric,
            }
            
            torch.save(checkpoint, save_path)
            
            # Log model as artifact to wandb
            if self.params.use_weights_and_biases:
                artifact = wandb.Artifact(f"best_model_epoch_{self.epoch}", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

            logger.info(f"Best model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def save_model_artifact(self):
        try:
            save_dir = os.path.join(self.params.checkpoint_file_folder, self.params.wandb_run_name)
            save_path = os.path.join(save_dir, f'best_model_epoch_{self.epoch}.pth')
            
            if os.path.isfile(save_path):
                artifact = wandb.Artifact(f"model_{self.epoch}", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
                logging.info(f"Model artifact saved for epoch {self.epoch}")
            else:
                logging.error(f"Model file not found: {save_path}")
        except Exception as e:
            logging.error(f"Error saving model artifact: {str(e)}")

    def main(self):
        try:
            logging.info("Training with the following parameters:")
            self.train(self.starting_epoch, self.params.max_epochs)
            if self.params.use_weights_and_biases:
                self.save_model_artifact()
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
        finally:
            if self.params.use_weights_and_biases:
                try:
                    wandb.finish()
                except Exception as e:
                    logging.error(f"Error finishing wandb run: {str(e)}")
                 

def main():
    args_parser = ArgsParser()
    args_parser()
    trainer_params = args_parser.arguments

    trainer = Trainer(trainer_params)
    trainer.main()
    
if __name__=="__main__":
    main()