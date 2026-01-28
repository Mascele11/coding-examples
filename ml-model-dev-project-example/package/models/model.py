# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd


# ======================================================================================================================
#   Abstract Class
# ======================================================================================================================
class Model(object, metaclass=ABCMeta):

    # ------- methods ----------------------------------------------------------
    @abstractmethod
    def train(self, data: Union[np.ndarray, pd.DataFrame], target: Union[np.ndarray, pd.Series] = None,
              seed: int = None) -> 'Model':
        """
        Train the model over the provided data and return the updated object.
        The random seed allows reproducibility, if not set it is randomly initialized.

        Parameters
        ----------
        data : Numpy 2D array or Pandas dataframe
            Training set of data samples.
        target: Numpy 1D array or Pandas series
            Ground-truth of the training set.
        seed : int, optional
            Random seed to initialize the training at the same conditions.

        Returns
        -------
        model : scikit-learn model or Keras model or PyTorch model
            Provide the current object itself, trained over the provided data.
        """
        return

    @abstractmethod
    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        """
        Exploit the model in inference.

        Parameters
        ----------
        data : Numpy array or Pandas dataframe
            Testing set.

        Returns
        -------
        results : Pandas dataframe
            Dataframe containing a row for each input sample. The minimum set of columns is
            'categorical' (one-hot encoded) and 'class' (corresponding index).
            Optional columns are 'distribution' (softmax probability distribution) and 'label' (human-readable class name).
        """
        return

    @abstractmethod
    def load(self, model_path: Union[str, Path]) -> Any:
        """
        Load a model stored on the local or remote file system.

        Parameters
        ----------
        model_path : string or Path
            Filename of the model persistent version.

        Returns
        -------
        model : scikit-learn model or Keras model or PyTorch model
            Provide the current object itself.
        """
        return

    @abstractmethod
    def save(self, model_path: Union[str, Path]):
        """
        Store the model on a local or remote file system.

        Parameters
        ----------
        model_path : string or Path
            Filename of the model persistent location where to store it.
        """
        return
