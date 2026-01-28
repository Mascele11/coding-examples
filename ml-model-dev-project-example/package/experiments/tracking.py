# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from abc import ABCMeta, abstractmethod
from typing import Any
from datetime import datetime
from dataclasses import dataclass


# ======================================================================================================================
#   Abstract Class
# ======================================================================================================================
@dataclass(init=False)
class ExperimentTracking(object, metaclass=ABCMeta):
    # ------- attributes -------------------------------------------------------
    # ------- public -------
    id: str = None
    is_active: bool = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, id: str = None,
                 is_active: bool = True):
        """ The creation of the object is considered as the start of the experiment """
        # identifier
        self.id = id if id is not None else datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # timestamp as unique ID

        # check if tracking is active
        self.is_active = is_active

    # ------- methods ----------------------------------------------------------
    @abstractmethod
    def register_parameter(self, key: str, value: Any):
        """ """
        return

    @abstractmethod
    def register_result(self, key: str, value: Any):
        """ """
        return

    @abstractmethod
    def already_exists(self, id: str) -> bool:
        """ """
        return
