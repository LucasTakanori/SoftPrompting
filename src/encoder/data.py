import csv
from torch.utils.data import Dataset
import logging 
import copy
import json
import torchaudio
import torch.nn.functional as F
from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim, load_audio
import torch
import random
from random import randint
import numpy as np
from whisper.tokenizer import get_tokenizer
from typing import List, Optional, Tuple, AnyStr
import re
from torch.nn.utils.rnn import pad_sequence



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
    def __init__(self, utterances_paths, processor, random_crop_secs, tokens_max_length, 
                 speech_representation, prompt_use_rate, prompt_length, vocab_size, 
                 nmels=80, padding_type="zero_pad", augmentation_prob=0, sample_rate=16000):
        self.utterances_paths = utterances_paths
        self.processor = processor
        self.random_crop_secs = random_crop_secs
        self.tokens_max_length = tokens_max_length
        self.speech_representation = speech_representation
        self.prompt_use_rate = prompt_use_rate
        self.prompt_length = prompt_length
        self.vocab_size = vocab_size
        self.nmels = nmels
        self.padding_type = padding_type
        self.augmentation_prob = augmentation_prob
        self.sample_rate = sample_rate
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.read_tsv()

    def read_json(self):
        self.utterances = []
        with open(self.utterances_paths, 'r') as f:
            for line in f:
                self.utterances.append(json.loads(line)) 
        
    def read_tsv(self):
        self.utterances = []
        with open(self.utterances_paths, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                self.utterances.append(row)
        
    def __len__(self):
        return len(self.utterances)

    #region transcription
    def init_tokenizer(self):
        logger.info(f"Initializing tokenizer")
        self.tokenizer = get_tokenizer(self.whisper_flavour)

    def get_transcription_tokens(self, transcription):
        indexed_tokens = self.tokenizer.encode(transcription)
        #tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = torch.tensor(indexed_tokens)
        return tokens_tensor

    def pad_transcription(self, transcription_tokens):
        """
        Pads the transcription tokens with zeros to match the maximum length.
        """
        pad_left = max(0, self.tokens_max_length - transcription_tokens.shape[-1])
        #logger.info(f"In the file data.py and function pad_Transcription() padding added to transcription: {pad_left}")
        padded_transcription_tokens = torch.nn.functional.pad(transcription_tokens, (pad_left, 0), mode = "constant")

        return padded_transcription_tokens
    #endregion

    #region waveform
    def init_data_augmentator(self):
        #TODO: Implement data augmentator
        ...


    def normalize(self, waveform):
        # Normalize the waveform to have zero mean and unit variance
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        return waveform

    def pad_waveform(self, waveform, padding_type, random_crop_samples):
        """
        If the waveform is shorter than the window, we make padding to allow cropping longer segments.

        Two padding systems: repetition padding which literally repeats the waveform until it reaches the desired length 
        and zero padding which adds zeros to the left of the waveform until it reaches the desired length.
        """
        if padding_type == "zero_pad":
            pad_left = max(0, self.random_crop_samples - waveform.shape[-1]) #HACK
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
    
    #endregion

    #region utterance

    def _calculate_mel(self, audio_path: str) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        ## insert some zero frames to hold the places for prompts
        n_frames = (1 + self.context_len) * 2  # 1 speaker embedding + 8 soft prompts, the down-sampling factor is 2 in encoder conv layers
        place_holder = torch.zeros((mel.size(0), n_frames))
        mel = torch.concat([place_holder, mel], dim=1)
        ###
        mel = pad_or_trim(mel, N_FRAMES-100) #HACK HACK HACK HACK 
        return mel
    
    def process_utterance(self, waveform):
        """
        Processing the waveform to create speech representations.
        """
        if self.speech_representation == "waveform":
            # The trivial is to return this waveform.
            return waveform

        if self.speech_representation == "mel":
            mel = self._calculate_mel(waveform)
            return mel
            
        else:
            raise Exception("No speech representation found.")
    
    #endregion

    #region decoder input
    # TODO: Understand this.

    def _encode_text_with_timestamps(self, text: str) -> List[int]:
        """
        Encodes the given text with timestamps into a list of tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            List[int]: The list of encoded tokens.

        Raises:
            ValueError: If an invalid timestamp is encountered.

        """
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])

                # timestamp must be in the range [0, 30] and be a multiple of 0.02 seconds
                if timestamp < 0 or timestamp > 30 or round(timestamp * 100) % 2 != 0:
                    raise ValueError(f"Invalid timestamp: {timestamp}")

                token = self.tokenizer.timestamp_begin + round(timestamp * 100) // 2
                tokens.append(token)
            else:
                tokens.extend(self.tokenizer.encode(part))

        return tokens
    def _get_partial_segment_start(self, tokens: List[int]) -> Optional[float]:
        if (
            len(tokens) >= 2
            and tokens[-2] >= self.tokenizer.timestamp_begin
            and tokens[-1] >= self.tokenizer.timestamp_begin
        ):  # if the last token is a start time token
            return (tokens[-1] - self.tokenizer.timestamp_begin) * 0.02
        else:
            return None

    def _get_text_tokens(self, text: str, no_timestamps: bool) -> Tuple[List[int], Optional[float]]:
        text_tokens = self._encode_text_with_timestamps(text)
        next_partial_segment_start = self._get_partial_segment_start(text_tokens)
        if no_timestamps:
            text_tokens = list(filter(lambda x: x < self.tokenizer.timestamp_begin, text_tokens))

        return text_tokens, next_partial_segment_start
    
    def _get_prompt_tokens(self, prompt: str) -> List[int]:
        if len(prompt) > 0 and torch.rand(1) < self.prompt_use_rate:
            prompt_tokens = self._encode_text_with_timestamps(prompt)[-self.prompt_length :]
            prompt_tokens = [self.tokenizer.sot_prev] + prompt_tokens
        else:
            prompt_tokens = []

        return prompt_tokens

    def _get_special_tokens(
        self, is_text_empty: bool, language: str, no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                special_tokens.append(self.tokenizer.no_timestamps)

        return special_tokens
    
    #endregion
    
    #region ground_truth
    def _construct_ground_truth(
        self, prompt_tokens: List[int], special_tokens: List[int], text_tokens: List[int]
    ) -> List[int]:
        if len(prompt_tokens) == 0:
            ground_truth = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        else:
            ground_truth = (
                # Mask out the training loss for predicting the prompt tokens. We use "-100" as the
                # default value for the `ignore_index` parameter in
                # `torch.nn.functional.cross_entropy()`. However, we do not mask out the loss for
                # predicting the sot token because our experiment indicates that the original
                # Whisper model assigns a high probability to the sot token after prompt tokens.
                [-100] * (len(prompt_tokens) - 1)
                + special_tokens
                + text_tokens
                + [self.tokenizer.eot]
            )
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)
        return ground_truth
    #endregion


    def __getitem__(self, index):
        utterance_path = self.utterances[index]["audio_path"]
        transcription = self.utterances[index]["text"]

        # Load and preprocess audio
        audio_input, sr = torchaudio.load(utterance_path)
        audio_input = self.process_waveform(audio_input, sr)
        
        # Process audio input
        input_features = self.processor(audio_input, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        input_features = input_features.squeeze(0)  # Remove batch dimension
        
        # Pad or trim input features to account for soft prompts
        target_length =  N_FRAMES  # N_FRAMES should be defined based on Whisper's requirements
        if input_features.shape[1] < target_length:
            input_features = F.pad(input_features, (0, target_length - input_features.shape[1]))
        else:
            input_features = input_features[:, :target_length]

        # Tokenize text
        labels = self.processor.tokenizer(transcription, return_tensors="pt").input_ids.squeeze(0)
        
        # Pad or trim labels to the specified max token length
        if labels.shape[0] < self.tokens_max_length:
            labels = F.pad(labels, (0, self.tokens_max_length - labels.shape[0]), value=self.processor.tokenizer.pad_token_id)
        else:
            labels = labels[:self.tokens_max_length]

        return input_features, labels    
