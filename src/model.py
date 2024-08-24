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
        
        # New parameter to determine where to apply soft prompts
        self.soft_prompt_location = parameters.soft_prompt_location

    def forward(self, input_features, labels=None):
        if self.soft_prompt_location == "encoder":
            prompted_features = self.soft_prompting(input_features)
            decoder_inputs = None
        elif self.soft_prompt_location == "decoder":
            prompted_features = input_features
            decoder_inputs = self.prepare_decoder_input(labels)
        else:
            raise ValueError("Invalid soft_prompt_location. Choose 'encoder' or 'decoder'.")
        
        if self.training:
            # During training, we use the standard forward pass
            outputs = self.whisper(input_features=prompted_features, labels=labels, decoder_inputs_embeds=decoder_inputs)
        else:
            # During inference, we use the generate method
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ca", task="transcribe")
            generated_ids = self.whisper.generate(
                inputs=prompted_features,
                forced_decoder_ids=forced_decoder_ids,
                decoder_inputs_embeds=decoder_inputs
            )
            outputs = {"generated_ids": generated_ids}

        return outputs

    def prepare_decoder_input(self, labels):
        if labels is None:
            return None
        
        soft_prompt = self.soft_prompting.soft_prompt.expand(labels.size(0), -1, -1)
        decoder_inputs = self.whisper.decoder.embed_tokens(labels)
        
        # Concatenate soft prompt with decoder input embeddings
        decoder_inputs = torch.cat([soft_prompt, decoder_inputs], dim=1)
        
        return decoder_inputs