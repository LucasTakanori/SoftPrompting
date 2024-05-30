import logging
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
    def __init__(self, parameters):
        self.parameters = parameters
        self.init_whisper()

    def init_whisper(self):
        self.select_whisper()

        # Freeze all parameters
        for name, parameter in self.asr.named_parameters():
            logger.info(f"Parameter {name} requires grad: False")
            parameter.requires_grad = False

    def select_whisper(self):
        self.asr = whisper.load_model(self.parameters.whisper_flavour, self.parameters.device)
        self.tokenizer = get_tokenizer(self.parameters.whisper_flavour)

    def run_whisper(self, input_tensor, soft_prompts):
        logits = self.asr(input_tensor, soft_prompts)
        return logits
    
    def __call__(self, input_tensor, soft_prompts):
        return self.run_whisper(input_tensor, soft_prompts)