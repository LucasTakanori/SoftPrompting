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
        self.params = parameters
        self.init_asr()
        self.init_soft_prompting()
    
    def init_asr(self):
        if self.params.asr_model == 'whisper':
            self.asr = Whisper(self.params, self.device)
        
    def init_soft_prompting(self):
        self.soft_prompting = SoftPrompting(self.params)

    # def forward(self, input_tensor, decoder_input) -> torch.Tensor:
    #     logger.info(f"In file model.py and function forward() The input tensor at the beginning shape is {input_tensor.shape}")
    #     logits = self.asr(input_tensor, decoder_input, self.soft_prompting())   

    #     return logits
    
    def forward(self, input_tensor, decoder_input) -> torch.Tensor:
            logger.info(f"In file model.py and function forward() The input tensor at the beginning shape is {input_tensor.shape}")
            # Get the soft prompts
            #soft_prompts = self.soft_prompting()
            # Concatenate the soft prompts with the input tensor along the last dimension (feature dimension)
            #enhanced_input = torch.cat([input_tensor, soft_prompts], dim=2)
            # Log the shapes for debugging
            logger.info(f"Enhanced input shape after concatenation: {input_tensor.shape}")
            # Pass the enhanced input to the ASR model
            logits = self.asr(input_tensor, decoder_input,self.soft_prompting.get_tensor())   
            return logits
    
    
