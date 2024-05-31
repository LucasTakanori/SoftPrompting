from torch import nn
import torch

class SoftPrompting(nn.Module):
    def __init__(self, parameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.soft_prompt_encoder = nn.Parameter(torch.Tensor(self.parameters.prompt_depth, self.parameters.prompt_length, self.parameters.prompt_dim),
                                                 requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_encoder)        
    
    def forward(self):

        return self.soft_prompt_encoder