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
            parameters.prompt_length,
            self.whisper.config.d_model,
            parameters.soft_prompt_location
        ).to(device)
        
        # Freeze the Whisper model
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Only the soft prompt parameters should be trainable
        self.trainable_params = list(self.soft_prompting.parameters())
        
        self.soft_prompt_location = parameters.soft_prompt_location
        
        # Set up forced decoder IDs for Catalan transcription
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ca", task="transcribe")

    def forward(self, input_features, labels=None):
        # Apply encoder soft prompts if location is "encoder" or "both"
        if self.soft_prompt_location in ["encoder", "both"]:
            prompted_features = self.soft_prompting.forward_encoder(input_features)
        else:
            prompted_features = input_features
        
        if self.training:
            # During training, we use the standard forward pass
            if self.soft_prompt_location in ["decoder", "both"]:
                # For decoder prompts, we need to run the encoder first
                encoder_outputs = self.whisper.get_encoder()(prompted_features)
                decoder_inputs = self.whisper.get_decoder().embed_tokens(labels)
                decoder_inputs = self.soft_prompting.forward_decoder(decoder_inputs)
                outputs = self.whisper(
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=decoder_inputs,
                    labels=labels
                )
            else:
                outputs = self.whisper(
                    input_features=prompted_features,
                    labels=labels
                )
        else:
            # During inference, we use the generate method
            if self.soft_prompt_location in ["decoder", "both"]:
                # Custom generation function to incorporate decoder soft prompts
                def custom_generate(**kwargs):
                    encoder_outputs = self.whisper.get_encoder()(kwargs['inputs'])
                    decoder_start_token = torch.full((kwargs['inputs'].size(0), 1), self.whisper.config.decoder_start_token_id, device=self.device)
                    decoder_inputs = self.whisper.get_decoder().embed_tokens(decoder_start_token)
                    decoder_inputs = self.soft_prompting.forward_decoder(decoder_inputs)
                    return self.whisper.generate(
                        encoder_outputs=encoder_outputs,
                        decoder_inputs_embeds=decoder_inputs,
                        forced_decoder_ids=self.forced_decoder_ids
                    )
                
                generated_ids = custom_generate(inputs=prompted_features)
                
                # Remove the soft prompt tokens from the output
                generated_ids = generated_ids[:, self.soft_prompting.prompt_length:]
            else:
                generated_ids = self.whisper.generate(
                    inputs=prompted_features,
                    forced_decoder_ids=self.forced_decoder_ids
                )
            
            outputs = {"generated_ids": generated_ids}

        return outputs