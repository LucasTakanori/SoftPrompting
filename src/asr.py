import logging
import torch
import whisper
from whisper.tokenizer import get_tokenizer

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


class Whisper():
    def __init__(self, parameters, device):
        self.parameters = parameters
        self.device = device
        self.init_whisper()

    def init_whisper(self):
        self.select_whisper()

        # Freeze all parameters
        for name, parameter in self.asr.named_parameters():
            logger.info(f"Parameter {name} requires grad: False")
            parameter.requires_grad = False

    def select_whisper(self):
        self.asr = whisper.load_model(self.parameters.whisper_flavour, self.device)
        self.tokenizer = get_tokenizer(self.parameters.whisper_flavour)

    def run_whisper(self, input_tensor, decoder_input, soft_prompts):
        input_concat = torch.cat((input_tensor, soft_prompts), dim=2)
        logger.info(f"Input_tensor shape: {input_tensor.shape}")
        logger.info(f"soft_prompts shape: {soft_prompts.shape}")
        logger.info(f"input_concat shape: {input_concat.shape}")
        logits = self.asr(input_concat, decoder_input)
        return logits
    
    def __call__(self, input_tensor, decoder_input,  soft_prompts):
        return self.run_whisper(input_tensor, decoder_input, soft_prompts)
