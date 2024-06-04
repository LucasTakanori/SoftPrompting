from torch import nn
import torch

class SoftPrompting(nn.Module):
    def __init__(self, parameters) -> None:
        super().__init__()
        self.parameters = parameters
        # The nmels parameter is 80 in the case of Whisper. 
        self.soft_prompt_encoder = nn.Parameter(torch.Tensor(self.parameters.batch_size, self.parameters.nmels ,self.parameters.prompt_length),
                                                 requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_encoder)        
    
    def forward(self):

        return self.soft_prompt_encoder