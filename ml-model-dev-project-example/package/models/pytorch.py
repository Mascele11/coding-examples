# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from pathlib import Path
from typing import Union, Any, Tuple

import numpy as np
import pandas as pd


import torch

# ------- custom modules -------
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
# file format
torch_extensions: [str] = ['.pth']


# ======================================================================================================================
#   Class
# ======================================================================================================================

class TemporalGNN(torch.nn.Module):
    # ------- attributes -------------------------------------------------------
    pytorch_model: torch.Module = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=600,
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(600, periods)

    # ------- methods ----------------------------------------------------------
    def forward(self, x, edge_index, edge_attributes):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(X=x, edge_index=edge_index, edge_weight=edge_attributes)
        h = F.relu(h)
        h = self.linear(h)
        return h

    # ------- internal facilities ----------------------------------------------
    def _architecture(self, input_shape: Tuple[int], output_shape: int,
                      optimizer: Union[str, Any] = 'adam', learning_rate: float = 1e-03) -> nn.Module:
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need
