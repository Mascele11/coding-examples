# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from pathlib import Path
from typing import Union, Any, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ------- custom modules -------
from sampleapplication.package.models.deep import DeepModel


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
# file format
torch_extensions: [str] = ['.pth']


# ======================================================================================================================
#   Class
# ======================================================================================================================
class PytorchModel(DeepModel):
    # ------- attributes -------------------------------------------------------
    pytorch_model: nn.Module = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, optimizer):
        return

    # ------- methods ----------------------------------------------------------
    def train(self, data: Union[np.ndarray, pd.DataFrame], target: Union[np.ndarray, pd.Series] = None,
              epochs: int = 5, batch_size: int = 32, seed: int = None) -> 'PytorchModel':
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need

    def load(self, model_path: Union[str, Path]) -> 'PytorchModel':
        model_path = Path(model_path)
        assert model_path.suffix in torch_extensions
        self.pytorch_model = torch.load(model_path)
        return self

    def save(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        assert model_path.suffix in torch_extensions
        torch.save(self.pytorch_model, model_path)

    # ------- internal facilities ----------------------------------------------
    def _architecture(self, input_shape: Tuple[int], output_shape: int,
                      optimizer: Union[str, Any] = 'adam', learning_rate: float = 1e-03) -> nn.Module:
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need
