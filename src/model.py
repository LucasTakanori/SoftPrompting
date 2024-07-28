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
        
        # Create decoder_input_ids
        batch_size = input_features.size(0)
        decoder_input_ids = torch.tensor([[self.processor.tokenizer.bos_token_id]] * batch_size).to(self.device)

        # If we are training (labels are provided), use teacher forcing
        if labels is not None:
            decoder_input_ids = torch.cat([decoder_input_ids, labels[:, :-1]], dim=1)

        outputs = self.whisper(
            input_features=prompted_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

        return outputs.logits