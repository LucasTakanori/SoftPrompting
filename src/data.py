from torch.utils.data import Dataset
import logging 
import copy
import json
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
        with open(self.utterances_paths, 'r') as f:
            for line in f:
                data = json.loads(line)
        print(data)
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        ...        

data = TrainDataset("/home/usuaris/veussd/lucas.takanori/lt400/lt400.json")