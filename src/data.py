from torch.utils.data import Dataset
import logging 
import copy
import json
import torchaudio
import torch
import random
#region Logging

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


class TrainDataset(Dataset):
    def __init__(self, utterances_paths, augmentation_prob = 0, sample_rate = 16000, waveforms_mean = None, waveforms_std = None):
        
        self.utterances_paths = utterances_paths
        # I suspect when instantiating two datasets the parameters are overrided
        self.augmentation_prob = augmentation_prob #TODO: implement data augmentation
        self.sample_rate = sample_rate
        self.waveforms_mean = waveforms_mean
        self.waveforms_std = waveforms_std
        self.num_samples = len(utterances_paths)
        self.read_json()
        if self.augmentation_prob > 0: self.init_data_augmentator()

    def read_json(self):
        self.utterances = []
        with open(self.utterances_paths, 'r') as f:
            for line in f:
                self.utterances.append(json.loads(line)) 
        

    def __len__(self):
        return self.num_samples

    def init_data_augmentator(self):
        #TODO: Implement data augmentator
        ...


    def normalize(self, waveform):

        if self.waveforms_mean is not None and self.waveforms_std is not None:
            normalized_waveform = (waveform - self.waveforms_mean) / (self.waveforms_std + 0.000001)
        else:
            normalized_waveform = waveform

        return normalized_waveform    


    def process_waveform(self, waveform: torch.Tensor, original_sample_rate:int):
        
        # Resample
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform = waveform,
                orig_freq = original_sample_rate, 
                new_freq = self.sample_rate, 
                )     
            
        # Normalize
        if self.waveforms_mean is not None:
            waveform = (waveform - self.waveforms_mean) / self.waveforms_std

        # Apply data augmentation if it falls within the probability
        if random.uniform(0, 0.999) > 1 - self.augmentation_prob:
            # TODO: Lucas, the data augmentator master will do this part. 
            waveform = self.data_augmentator(waveform, self.sample_rate)

        # stereo to mono
        waveform_mono = torch.mean(waveform, dim=0)
        waveform = waveform_mono.squeeze(0)

        waveform = self.normalize(waveform)

        return waveform

    def __getitem__(self, index):

        utterance_path = self.utterances[index]["audio_path"]
        transcription = self.utterances[index]["text"]

        waveform, initial_sample_rate = torchaudio.load(utterance_path)       
        
        waveform = self.process_waveform(waveform, initial_sample_rate)

        return waveform, transcription

