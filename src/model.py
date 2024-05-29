import logging
from torch import nn 
import torch

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

    def forward(self, input_tensor) -> torch.Tensor:
        return input_tensor