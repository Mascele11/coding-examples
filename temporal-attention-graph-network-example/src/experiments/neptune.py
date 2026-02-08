# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
import dataclasses
import time
import json
import yaml
import atexit
from pathlib import Path
from typing import Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum

import neptune.new as neptune
from neptune.new.run import Run

# ------- custom modules -------
from package.experiments.tracking import ExperimentTracking


# ======================================================================================================================
#   Class
# ======================================================================================================================
@dataclass(init=False)
class NeptuneExperimentTracking(ExperimentTracking):
    # ------- attributes -------------------------------------------------------
    # ------- public -------
    neptune_run: Run = None

    # ------- private -------
    _neptune_token: str = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, id: str = None,
                 is_active: bool = True,
                 workspace: str = None, project: str = None, token: str = None,
                 tags: [str] = [], description: str = ''):
        """ The creation of the object is considered as the start of the experiment """
        # initialize parent class
        super().__init__(id, is_active)

        # avoid non-required computations
        if not self.is_active:
            return

        # store token
        self._neptune_token = token

        # instantiate experiment
        self.neptune_run = neptune.init(
            project=f'{workspace}/{project}',
            name=id,
            tags=tags, description=description,
            source_files='*.py'  #TODO extend to the whole Python src
        )

    # ------- methods ----------------------------------------------------------
    def register_parameter(self, key: str, value: Any):
        """ """
        if not self.is_active:
            return

        key, value = str(key), str(value)  # serialize input
        nested_dict: dict = reduce(lambda res, cur: {cur: res}, reversed(key.split("/")), value)  # split / as dictionary levels
        self.neptune_run['params'] = self.neptune_run['params'] | nested_dict  # merge two dictionaries
        return

    def register_result(self, key: str, value: Any):
        """ """
        if not self.is_active:
            return

        key, value = str(key), str(value)  # serialize input
        nested_dict: dict = reduce(lambda res, cur: {cur: res}, reversed(key.split("/")), value)  # split / as dictionary levels
        self.neptune_run = self.neptune_run | nested_dict  # merge two dictionaries
        return
