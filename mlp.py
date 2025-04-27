import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple

class MLP(nn.Module):
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        output = input_data
        for layer in self.layers:
            output = layer(output)
        return output

    def __init__(self, layer_dimensions: Tuple[int, ...], include_bias=True, activation_func=nn.ReLU):
        super(MLP, self).__init__()
        
        # Create a list to hold layer definitions
        layer_list = []
        
        # Iterate over pairs of consecutive layer sizes to create linear layers
        for idx in range(1, len(layer_dimensions)):
            layer_list.append(nn.Linear(layer_dimensions[idx - 1], layer_dimensions[idx], bias=include_bias))
            
            # Add activation functions except after the last layer
            if idx < len(layer_dimensions) - 1:
                layer_list.append(activation_func())
        
        # Bundle layers into a Sequential container
        self.layers = nn.ModuleList(layer_list)