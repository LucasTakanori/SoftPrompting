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
    
    # def forward(self):

    #     return self.soft_prompt_encoder

    def forward(self, input_tensor):
        # Expanding soft_prompt_encoder to match the batch size of the input_tensor
        prompt = self.soft_prompt_encoder.expand(input_tensor.size(0), -1, -1)
        # Concatenating the prompt to the input tensor along the appropriate dimension
        enhanced_input = torch.cat([input_tensor, prompt], dim=2)  # Adjust the dimension as necessary
        return enhanced_input