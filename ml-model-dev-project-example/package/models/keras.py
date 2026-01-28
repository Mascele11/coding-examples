# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from pathlib import Path
from typing import Union, Any, Tuple

import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.callbacks import History

# ------- custom modules -------

# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
# file format
h5_extensions: [str] = ['.h5', '.hdf5', '.keras']  # old Keras H5 format
savedmodel_extensions: [str] = ['']  # SavedModel format is a directory


# ======================================================================================================================
#   Class
# ======================================================================================================================
class KerasModel(DeepModel):
    # ------- attributes -------------------------------------------------------
    keras_model: Model = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, input_shape: Tuple[int], output_shape: int,
                      optimizer: Union[str, Optimizer] = 'adam', learning_rate: float = 1e-03):
        self.keras_model = self._architecture(input_shape, output_shape, optimizer, learning_rate)
        return

    # ------- methods ----------------------------------------------------------
    def train(self, data: Union[np.ndarray, pd.DataFrame], target: Union[np.ndarray, pd.Series] = None,
              epochs: int = 5, batch_size: int = 32, seed: int = None) -> 'KerasModel':
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need

    def load(self, model_path: Union[str, Path]) -> 'KerasModel':
        model_path = Path(model_path)
        assert model_path.suffix in h5_extensions or model_path.suffix in savedmodel_extensions
        self.keras_model = keras.models.load_model(model_path)
        return self

    def save(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        assert model_path.suffix in h5_extensions or model_path.suffix in savedmodel_extensions
        self.keras_model.save(model_path)

    # ------- internal facilities ----------------------------------------------
    def _architecture(self, input_shape: Tuple[int], output_shape: int,
                      optimizer: Union[str, Optimizer] = 'adam', learning_rate: float = 1e-03) -> Model:
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your need
