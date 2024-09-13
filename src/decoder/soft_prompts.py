import torch
from torch import nn

class SoftPrompting(nn.Module):
    def __init__(self, decoder_dim, prompt_length):
        super().__init__()
        self.decoder_soft_prompt = nn.Parameter(torch.empty(1, prompt_length, decoder_dim))
        nn.init.xavier_uniform_(self.decoder_soft_prompt)
        self.prompt_length = prompt_length
    
    def forward(self, decoder_inputs):
        batch_size = decoder_inputs.size(0)
        expanded_prompt = self.decoder_soft_prompt.expand(batch_size, -1, -1)
        return torch.cat([expanded_prompt, decoder_inputs[:, self.prompt_length:, :]], dim=1)