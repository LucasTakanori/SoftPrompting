import torch
from torch import nn

class SoftPrompting(nn.Module):
    def __init__(self, num_mel_bins, prompt_length, decoder_dim, prompt_location):
        super().__init__()
        self.prompt_location = prompt_location
        
        if prompt_location in ["encoder", "both"]:
            self.encoder_soft_prompt = nn.Parameter(torch.empty(1, num_mel_bins, prompt_length))
            nn.init.xavier_uniform_(self.encoder_soft_prompt)
        
        if prompt_location in ["decoder", "both"]:
            self.decoder_soft_prompt = nn.Parameter(torch.empty(1, prompt_length, decoder_dim))
            nn.init.xavier_uniform_(self.decoder_soft_prompt)
        
        self.prompt_length = prompt_length
    
    def forward_encoder(self, input_features):
        if self.prompt_location in ["encoder", "both"]:
            batch_size = input_features.size(0)
            expanded_prompt = self.encoder_soft_prompt.expand(batch_size, -1, -1)
            return torch.cat([expanded_prompt, input_features[:, :, self.prompt_length:]], dim=2)
        return input_features
    
    def forward_decoder(self, decoder_inputs):
        if self.prompt_location in ["decoder", "both"]:
            batch_size = decoder_inputs.size(0)
            expanded_prompt = self.decoder_soft_prompt.expand(batch_size, -1, -1)
            return torch.cat([expanded_prompt, decoder_inputs[:, self.prompt_length:, :]], dim=1)
        return decoder_inputs