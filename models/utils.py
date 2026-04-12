import torch.nn as nn

def get_activation(activation: str):

    if activation is None:
        act = None
    elif activation == 'gelu':
        act = nn.GELU()            
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'tanh':
        act = nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return act