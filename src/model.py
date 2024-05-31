import logging
from torch import nn 
import torch
from whisper.tokenizer import get_tokenizer
from soft_prompts import SoftPrompting
from asr import Whisper

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

class PromptASR(nn.Module):
    def __init__(self, parameters, device) -> None:
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.init_asr()
        self.init_soft_prompting()
    
    def init_asr(self):
        if self.parameters.asr_model == 'whisper':
            self.asr = Whisper(self.parameters, self.device)
        
    def init_soft_prompting(self):
        self.soft_prompting = SoftPrompting(self.parameters)

    def forward(self, input_tensor) -> torch.Tensor:
        logger.info(f"The input tensor at the beginning shape is {input_tensor.shape}")
        logits = self.asr(input_tensor, self.soft_prompting())   
        return logits
    
    
    
