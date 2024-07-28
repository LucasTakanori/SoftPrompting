import torch
from torch import nn

class SoftPrompting(nn.Module):
    def __init__(self, batch_size, num_mel_bins, prompt_length):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(batch_size, num_mel_bins, prompt_length))
    
    def forward(self, input_features):
        return torch.cat([self.soft_prompt, input_features], dim=-1)


    # def forward(self, input_tensor):
    #     # Expanding soft_prompt_encoder to match the batch size of the input_tensor
    #     #prompt = self.soft_prompt_encoder.expand(input_tensor.size(0), input_tensor.size(1), -1)
    #     enhanced_input = torch.cat([input_tensor, self.prompt], dim=2)  # Concatenate along the feature dimension
    #     return enhanced_input