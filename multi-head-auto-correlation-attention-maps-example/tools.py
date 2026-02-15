
from __future__ import annotations

import numpy as np
import pandas as pd

# Underlying implementations
from model.local_model_predictor import (
    InferenceInterface,
)

import config



class ViTimePredictor:
    """Thin wrapper around the underlying inference interface.

    Initializes model weights from `config.VITIME_MODEL_PATH` and exposes
    a callable that maps a time series and `future_length` to predictions.
    """


    def __init__(
        self,
        device: str = 'cuda:0',
        model_name: str = 'MAE',
        tempature=1,

    ) -> None:
        
        model_path_env = config.VITIME_MODEL_PATH  
        self.tempature=tempature
        self._iface = InferenceInterface(model_path_env,  model_name=model_name, device=device)

    def __call__(self, time_series, future_length,sampleNumber) -> np.ndarray:
      
        pred = self._iface.inference(
            np.asarray(time_series),
            future_length,
            sampleNumber,
            tempature=self.tempature
        )
        return pred


class ViTimePrediction():
    def __init__(self, device='cuda:0', model_name='MAE', lookbackRatio=1, tempature=1):
        """
        Initialize the ViTime predictor.

        Args:
            device (str): Compute device (e.g., 'cuda:0' or 'cpu').
            model_name (str): Model name to select backbone/weights.
            lookbackRatio (float): Fixed lookback ratio when not adaptive.

        """

        self.lookbackRatio = lookbackRatio
        self.predictor = ViTimePredictor(device=device, model_name=model_name, tempature=tempature)

    def prediction(self, historical_data, future_length, sampleNumber=None):
        '''
        historical_data: n-dimensional numpy array (T[, C]).
        Returns an array of length `future_length`.
        if sampleNumber is None, Output a deterministic prediction; if not, switch to the probabilistic prediction mode and indicate the number of samples.
        '''

        historical_length_orig = historical_data.shape[0]

        # Apply lookbackRatio to crop history
        if self.lookbackRatio is not None:
            lookback_len = int(future_length * self.lookbackRatio)
        else:
            lookback_len = historical_length_orig
        # Ensure we do not exceed original history length
        lookback_len = min(lookback_len, historical_length_orig)

        if lookback_len > 0:
            historical_data = historical_data[-lookback_len:]

        predictor = self.predictor
        full_prediction = predictor(historical_data, future_length, sampleNumber=sampleNumber)[:, 0]

        prediction = full_prediction[len(historical_data):len(historical_data) + future_length]

        # Step 3: check and linearly impute NaNs in output
        if np.isnan(np.sum(prediction)):
            s = pd.Series(prediction)
            s.interpolate(method='linear', limit_direction='both', inplace=True)
            prediction = s.to_numpy()

        return prediction
