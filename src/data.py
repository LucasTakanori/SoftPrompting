from torch.utils.data import Dataset
import logging 
import copy
import json
import torchaudio
from whisper.audio import CHUNK_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim, load_audio
import torch
import random
from random import randint
import numpy as np
from whisper.tokenizer import get_tokenizer
from typing import List, Optional, Tuple, AnyStr
import re


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

    """
    The main goal of this class is, given an index, to provide the audio utterance and transcription of the audio.

    However, because of the encoder-decoder architectures, we need to provide the decoder input as well.

    In the case of Whisper, the decoder input is the concatenation of the special tokens, the transcription tokens. 

    Focus on the __getitem__ method to understand the process and the needs of the problem.
    """
    def __init__(self, utterances_paths, whisper_flavour, random_crop_secs, context_len, tokens_max_length, speech_representation, prompt_use_rate, max_prompt_length, nmels=80, padding_type ="zero_pad", augmentation_prob = 0, sample_rate = 16000, waveforms_mean = None, waveforms_std = None):
        
        self.utterances_paths = utterances_paths
        # I suspect when instantiating two datasets the parameters are overrided
        self.augmentation_prob = augmentation_prob #TODO: implement data augmentation
        self.random_crop_secs = random_crop_secs
        self.speech_representation = speech_representation
        self.nmels = nmels
        self.language = "ca" # HACK whisper hardcoded
        self.context_len = context_len
        self.max_prompt_length = max_prompt_length
        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH # HACK whisper hardcoded
        self.whisper_flavour = whisper_flavour
        self.init_tokenizer()
        self.timestamp_pattern = re.compile(r"(<\|[123]?[0-9]\.[0-9][0-9]\|>)")
        self.padding_type = padding_type
        self.prompt_use_rate = prompt_use_rate
        self.sample_rate = sample_rate
        self.random_crop_samples = int(self.random_crop_secs * self.sample_rate)
        self.tokens_max_length = tokens_max_length
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

    #region transcription
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

    def pad_transcription(self, transcription_tokens):
        """
        Pads the transcription tokens with zeros to match the maximum length.
        """
        pad_left = max(0, self.tokens_max_length - transcription_tokens.shape[-1])
        padded_transcription_tokens = torch.nn.functional.pad(transcription_tokens, (pad_left, 0), mode = "constant")

        return padded_transcription_tokens
    #endregion

    #region waveform
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
    
    #endregion

    #region utterance

    def _calculate_mel(self, audio_path: str) -> torch.Tensor:
        mel = log_mel_spectrogram(audio_path)
        ## insert some zero frames to hold the places for prompts
        n_frames = (1 + self.context_len) * 2  # 1 speaker embedding + 8 soft prompts, the down-sampling factor is 2 in encoder conv layers
        place_holder = torch.zeros((mel.size(0), n_frames))
        mel = torch.concat([place_holder, mel], dim=1)
        ###
        mel = pad_or_trim(mel, N_FRAMES)
        
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
            prompt_tokens = self._encode_text_with_timestamps(prompt)[-self.max_prompt_length :]
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
    

    def __getitem__(self, index):
        
        # We get the waveform and the transcription:
        utterance_path = self.utterances[index]["audio_path"]
        transcription = self.utterances[index]["text"]

        # waveform modifications
        waveform, initial_sample_rate = torchaudio.load(utterance_path)       
        waveform = self.process_waveform(waveform, initial_sample_rate)

        # tokenizing transcription:
        transcription_tokens = self.get_transcription_tokens(transcription)
        transcription_tokens = self.pad_transcription(transcription_tokens)

        # decoder input
        # HACK will change later. For now, we are not using timestamps
        no_timestamps = True
        prompt_tokens = self._get_prompt_tokens('@' * (self.context_len))  # hole the place where will be filled with speaker embedding.
        text_tokens, next_partial_segment_start = self._get_text_tokens(transcription.lower(), no_timestamps)
        is_text_empty = len(text_tokens) == 0        
        special_tokens = self._get_special_tokens(is_text_empty, self.language, no_timestamps)
        # list with all the input of the decoder
        decoder_input =  prompt_tokens + special_tokens + text_tokens
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)


        # change to speech representation (ie mel-spectrogram)
        utterance = self.process_utterance(waveform)

        logger.info(f"shapes {utterance.shape}, {transcription_tokens.shape}, {decoder_input.shape}")
        return utterance, transcription_tokens, decoder_input

