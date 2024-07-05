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
        self.params = parameters
        self.device = device
        self.tokens_max_length = self.params.tokens_max_length
        self.init_whisper()

    def init_whisper(self):
        self.select_whisper()

        # Freeze all parameters
        for name, parameter in self.asr.named_parameters():
            logger.info(f"Parameter {name} requires grad: False")
            parameter.requires_grad = False

    def select_whisper(self):
        self.asr = whisper.load_model(self.params.whisper_flavour, self.device)
        self.decoding_options = whisper.DecodingOptions(language="ca", without_timestamps=True)# HACK Use parameters but for now test with this
        print(self.decoding_options)
        self.tokenizer = get_tokenizer(self.params.whisper_flavour)


    def pad_transcription(self, transcription_tokens):
        """
        Pads the transcription tokens with zeros to match the maximum length.
        """
        logger.info(f" tokens_max_length: {self.tokens_max_length}")
        pad_left = max(0, self.tokens_max_length - transcription_tokens.shape[-1])
        #logger.info(f"In the file data.py and function pad_Transcription() padding added to transcription: {pad_left}")
        padded_transcription_tokens = torch.nn.functional.pad(transcription_tokens, (pad_left, 0), mode = "constant")

        return padded_transcription_tokens
    
    def run_whisper(self, input_tensor, decoder_input, soft_prompt):
        # logger.info(            
        #     f"In File asr.py and function run_whisper():\n  Input_tensor shape: {input_tensor.shape}\n "
        # )
        # logger.info(
        #     f"In File asr.py and function run_whisper() decoder_input: {decoder_input.shape}"
        # )

        input_concat = torch.cat((input_tensor, soft_prompt), dim=2)
        print(input_concat.shape)
        
        results = self.asr.decode(input_concat ,self.decoding_options)
        

        tokens_list = [result.tokens for result in results]
        #print(tokens_list)
        #print(f"Decode output: {result}")

        tokens_tensor = torch.FloatTensor(tokens_list)
        tokens_padded = self.pad_transcription(tokens_tensor)

        return tokens_padded
        #logits = self.asr(input_concat, decoder_input)
        #return logits
    
    def __call__(self, input_tensor, decoder_input, soft_prompt):
        return self.run_whisper(input_tensor, decoder_input,soft_prompt)