import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from soft_prompts import SoftPrompting

class PromptASR(nn.Module):
    def __init__(self, parameters, device):
        super().__init__()
        self.device = device
        self.params = parameters
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.soft_prompting = SoftPrompting(parameters.batch_size, 
                                            self.whisper.config.num_mel_bins, 
                                            parameters.prompt_length).to(device)
        # Freeze the Whisper model
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Only the soft prompt parameters should be trainable
        self.trainable_params = list(self.soft_prompting.parameters())

    def forward(self, input_features):
        batch_size = input_features.shape[0]
        soft_prompts = self.soft_prompting(batch_size)
        prompted_features = torch.cat([soft_prompts, input_features], dim=1)
        return self.whisper(input_features=prompted_features).logits