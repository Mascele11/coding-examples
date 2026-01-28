# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
import logging
from logging import Logger

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
# setup logging
logger: Logger = logging.getLogger(__name__)


# ======================================================================================================================
#   Class
# ======================================================================================================================
class Preprocessing(BaseEstimator, TransformerMixin):
    # ------- attributes -------------------------------------------------------
    # TODO: insert persistent status

    # ------- constructor ------------------------------------------------------
    def __init__(self):
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs

    # ------- scikit-learn methods ---------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Preprocessing':
        # prepare the ground to provide the samples in the transform()
        logger.debug(f"Fit pre-processing model over {X.shape} data")
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        # actually pre-process data
        logger.debug(f"Preprocess {X.shape} data")
        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs
        return X
