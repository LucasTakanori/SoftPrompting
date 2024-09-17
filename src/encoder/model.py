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
        encoder_outputs = self.whisper.get_encoder()(input_features)
        
        if self.training:
            # During training
            # Add soft prompt to the beginning of decoder_input_ids
            soft_prompt = self.soft_prompting.decoder_soft_prompt.expand(decoder_input_ids.shape[0], -1, -1)
            decoder_inputs = self.whisper.get_decoder().embed_tokens(decoder_input_ids)
            decoder_inputs = torch.cat([soft_prompt, decoder_inputs], dim=1)
            
            outputs = self.whisper(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=decoder_inputs,
                labels=labels
            )
            
            # Generate text predictions for WER calculation
            with torch.no_grad():
                generated_ids = self.whisper.generate(
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=soft_prompt,  # Use only the soft prompt as initial input
                    forced_decoder_ids=self.forced_decoder_ids,
                    max_length=self.params.tokens_max_length
                )
            outputs["generated_ids"] = generated_ids
        else:
            # During inference
            soft_prompt = self.soft_prompting.decoder_soft_prompt.expand(input_features.size(0), -1, -1)
            
            generated_ids = self.whisper.generate(
                encoder_outputs=encoder_outputs,
                decoder_inputs_embeds=soft_prompt,
                forced_decoder_ids=self.forced_decoder_ids,
                max_length=self.params.tokens_max_length
            )
            
            outputs = {"generated_ids": generated_ids}

        return outputs