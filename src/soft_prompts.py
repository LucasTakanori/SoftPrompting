from torch import nn
import torch

class SoftPrompting(nn.Module):
    def __init__(self, parameters) -> None:
        super().__init__()
        #self.parameters = parameters
        # The nmels parameter is 80 in the case of Whisper. 
        self.soft_prompt_encoder = nn.Parameter(torch.Tensor(parameters.batch_size,
                                                              parameters.nmels,
                                                              parameters.prompt_length),
                                                              requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_encoder)        
    
    def get_tensor(self):

        return self.soft_prompt_encoder


    # def forward(self, input_tensor):
    #     # Expanding soft_prompt_encoder to match the batch size of the input_tensor
    #     #prompt = self.soft_prompt_encoder.expand(input_tensor.size(0), input_tensor.size(1), -1)
    #     enhanced_input = torch.cat([input_tensor, self.prompt], dim=2)  # Concatenate along the feature dimension
    #     return enhanced_input