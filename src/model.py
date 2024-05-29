import logging
from torch import nn 
import torch
import whisper
from whisper.tokenizer import get_tokenizer
from soft_prompts import SoftPrompting


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
    
    def select_whisper(self):
        self.model = whisper.load_model(self.parameters.whisper_flavour, self.parameters.device)
        self.tokenizer = get_tokenizer(self.parameters.whisper_flavour)
    
    def init_asr(self):
        self.select_whisper()
        for name, parameter in self.model.named_parameters():
            logger.info(f"Parameter {name} requires grad: False")
            parameter.requires_grad = False
        # self.model.eval()
    
    def init_soft_prompting(self):
        self.soft_prompting = SoftPrompting(self.parameters)

    def forward(self, input_tensor) -> torch.Tensor:
        
        logits = self.
        return logits
    
