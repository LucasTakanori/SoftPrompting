import torch
from torch import nn

class SoftPrompting(nn.Module):
    def __init__(self, num_mel_bins, prompt_length):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.empty(1, num_mel_bins, prompt_length))
        nn.init.xavier_uniform_(self.soft_prompt)
        self.prompt_length = prompt_length
    
    def forward(self, input_features):
        batch_size = input_features.size(0)
        expanded_prompt = self.soft_prompt.expand(batch_size, -1, -1)
        
        # Replace the beginning of input_features with the soft prompt
        prompted_features = torch.cat([expanded_prompt, input_features[:, :, self.prompt_length:]], dim=2)
        
        return prompted_features