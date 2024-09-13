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
            self.whisper.config.d_model,
            parameters.prompt_length
        ).to(device)
        
        # Freeze the Whisper model
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Only the soft prompt parameters should be trainable
        self.trainable_params = list(self.soft_prompting.parameters())
        
        # Set up forced decoder IDs for Catalan transcription
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ca", task="transcribe")
        
    def visualize_soft_prompts(self):
        prompt_weights = self.soft_prompting.decoder_soft_prompt.data
        return prompt_weights.cpu().numpy()


    def forward(self, input_features, decoder_input_ids=None, labels=None):
        if self.training:
            # During training, we use the standard forward pass
            encoder_outputs = self.whisper.get_encoder()(input_features)
            decoder_inputs = self.whisper.get_decoder().embed_tokens(decoder_input_ids)
            decoder_inputs = self.soft_prompting(decoder_inputs)
            outputs = self.whisper(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs,
                labels=labels
            )
        else:
            # During inference, we use the generate method
            encoder_outputs = self.whisper.get_encoder()(input_features)
            decoder_start_token = torch.full((input_features.size(0), 1), self.whisper.config.decoder_start_token_id, device=self.device)
            decoder_inputs = self.whisper.get_decoder().embed_tokens(decoder_start_token)
            decoder_inputs = self.soft_prompting(decoder_inputs)
            
            generated_ids = self.whisper.generate(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs,
                forced_decoder_ids=self.forced_decoder_ids,
                max_length=self.params.tokens_max_length
            )
            
            outputs = {"generated_ids": generated_ids}

        return outputs