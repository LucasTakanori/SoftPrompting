import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from soft_prompts import SoftPrompting

class PromptASR(nn.Module):
    def __init__(self, parameters, device):
        super().__init__()
        self.device = device
        self.params = parameters
        self.whisper = WhisperForConditionalGeneration.from_pretrained(self.params.whisper_flavour).to(device)
        self.processor = WhisperProcessor.from_pretrained(self.params.whisper_flavour)
        self.soft_prompting = SoftPrompting(
            self.whisper.config.num_mel_bins, 
            parameters.prompt_length
        ).to(device)
        
        # Freeze the Whisper model
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Only the soft prompt parameters should be trainable
        self.trainable_params = list(self.soft_prompting.parameters())

    def forward(self, input_features, labels=None):
        prompted_features = self.soft_prompting(input_features)
        
        if self.training:
            # During training, we use the standard forward pass
            outputs = self.whisper(input_features=prompted_features, labels=labels)
        else:
            # During inference, we use the generate method
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ca", task="transcribe")
            generated_ids = self.whisper.generate(
                inputs=prompted_features,
                forced_decoder_ids=forced_decoder_ids
            )
            outputs = {"generated_ids": generated_ids}

        return outputs