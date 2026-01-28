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
class AzureExperimentTracking(ExperimentTracking):
    # ------- attributes -------------------------------------------------------
    # ------- public -------
    resource_group: str = None
    aml_service: str = None

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

        # connect to the Azure Machine Learning service
        raise NotImplementedError("implement according to your need")

    # ------- methods ----------------------------------------------------------
    def register_parameter(self, key: str, value: Any):
        """ """
        # avoid non-required computations
        if not self.is_active:
            return

        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs

    def register_result(self, key: str, value: Any):
        """ """
        # avoid non-required computations
        if not self.is_active:
            return

        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs

    def already_exists(self, id: str) -> bool:
        """ """
        # avoid non-required computations
        if not self.is_active:
            return False

        raise NotImplementedError("implement according to your need")  # TODO: implement according to your needs
