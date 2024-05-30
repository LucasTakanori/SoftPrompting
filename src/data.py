from torch.utils.data import Dataset
import logging 
import copy
import json
import torchaudio
import torch
import random
from random import randint
import numpy as np
from whisper.tokenizer import get_tokenizer

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
    def __init__(self, utterances_paths, whisper_flavour, random_crop_secs, padding_type ="zero_pad", augmentation_prob = 0, sample_rate = 16000, waveforms_mean = None, waveforms_std = None):
        
        self.utterances_paths = utterances_paths
        # I suspect when instantiating two datasets the parameters are overrided
        self.augmentation_prob = augmentation_prob #TODO: implement data augmentation
        self.random_crop_secs = random_crop_secs
        self.whisper_flavour = whisper_flavour
        self.init_tokenizer()
        self.padding_type = padding_type
        self.sample_rate = sample_rate
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.waveforms_mean = waveforms_mean
        self.waveforms_std = waveforms_std
        self.read_json()
        if self.augmentation_prob > 0: self.init_data_augmentator()

    def read_json(self):
        self.utterances = []
        with open(self.utterances_paths, 'r') as f:
            for line in f:
                self.utterances.append(json.loads(line)) 
        

    def __len__(self):
        return len(self.utterances)

    def init_tokenizer(self):
        logger.info(f"Initializing tokenizer")

        if self.whisper_flavour == "medium":
            self.tokenizer = get_tokenizer(self.whisper_flavour)
        else:
            raise Exception("No tokenizer found for the specified flavour")


    def get_transcription_tokens(self, transcription):
        indexed_tokens = self.tokenizer.encode(transcription)
        #tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.tensor(indexed_tokens)
        return tokens_tensor
    

    def init_data_augmentator(self):
        #TODO: Implement data augmentator
        ...


    def normalize(self, waveform):

        if self.waveforms_mean is not None and self.waveforms_std is not None:
            normalized_waveform = (waveform - self.waveforms_mean) / (self.waveforms_std + 0.000001)
        else:
            normalized_waveform = waveform

        return normalized_waveform    

    def pad_waveform(self, waveform, padding_type, random_crop_samples):
        """
        If the waveform is shorter than the window, we make padding to allow cropping longer segments.

        Two padding systems: repetition padding which literally repeats the waveform until it reaches the desired length 
        and zero padding which adds zeros to the left of the waveform until it reaches the desired length.
        """
        if padding_type == "zero_pad":
            pad_left = max(0, self.random_crop_samples - waveform.shape[-1])
            padded_waveform = torch.nn.functional.pad(waveform, (pad_left, 0), mode = "constant")
        elif padding_type == "repetition_pad":
            necessary_repetitions = int(np.ceil(random_crop_samples / waveform.size(-1)))
            padded_waveform = waveform.repeat(necessary_repetitions)
        else:
            raise Exception('No padding choice found.') 
        
        return padded_waveform
    
    def sample_audio_window(self, waveform, random_crop_samples):

        waveform_total_samples = waveform.size()[-1]
        
        # TODO maybe we can add an assert to check that random_crop_samples <= waveform_total_samples (will it slow down the process?)
        random_start_index = randint(0, waveform_total_samples - random_crop_samples)
        end_index = random_start_index + random_crop_samples
        
        cropped_waveform =  waveform[random_start_index : end_index]

        return cropped_waveform

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

        if self.random_crop_secs > 0:
            # We make padding to allow cropping longer segments
            # (If not, we can only crop at most the duration of the shortest audio)
            if self.random_crop_samples > waveform.size(-1):
                waveform = self.pad_waveform(waveform, self.padding_type, self.random_crop_samples)
            
            # TODO torchaudio.load has frame_offset and num_frames params. Providing num_frames and frame_offset arguments is more efficient
            waveform = self.sample_audio_window(
                waveform, 
                random_crop_samples = self.random_crop_samples,
                )
        else:
            # HACK don't understand why, I have to do this slicing (which sample_audio_window does) to make dataloader work
            waveform =  waveform[:]


        waveform = self.normalize(waveform)

        return waveform

    def __getitem__(self, index):
        
        # We get the waveform and the transcription:
        utterance_path = self.utterances[index]["audio_path"]
        transcription = self.utterances[index]["text"]

        # waveform modifications
        waveform, initial_sample_rate = torchaudio.load(utterance_path)       
        waveform = self.process_waveform(waveform, initial_sample_rate)

        # tokenizing transcription:
        transcription_tokens = self.get_transcription_tokens(transcription)

        return waveform, transcription_tokens

