# Example File:

# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
import logging
from logging import Logger

import pandas as pd
import numpy as np

from typing import Union, List

import tensorflow as tf


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
# setup logging
logger: Logger = logging.getLogger(__name__)


# ======================================================================================================================
#   Class
# ======================================================================================================================
class DataGenerator(tf.keras.utils.Sequence):
    """
    Tutorial at:
    https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c
    """

    # ------- attributes -------------------------------------------------------
    # TODO: insert persistent status

    # ------- constructor ------------------------------------------------------
    def __init__(self, dataframe: pd.DataFrame, x_col: Union[str, List[str]], y_col: str = None,
                 batch_size: int = 32, num_classes: int = None, shuffle: bool = True):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    # ------- Keras callbacks --------------------------------------------------
    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    # ------- operator overloading ---------------------------------------------
    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    # ------- internal facilities ----------------------------------------------
    def __get_data(self, batch):
        X =  #TODO logic
        y =  #TODO logic

        for i, id in enumerate(batch):
            X[i,] =  #TODO logic
            y[i] =   #TODO labels

        return X, y
