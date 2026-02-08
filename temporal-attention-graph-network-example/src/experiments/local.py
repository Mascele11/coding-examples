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

import pandas as pd
from functools import reduce

# ------- custom modules -------
from package import PROJECT_ROOT
from package.experiments.tracking import ExperimentTracking
from package.experiments.platform import ComputingPlatform


# ======================================================================================================================
#   Types declaration
# ======================================================================================================================

class Execution(Enum):
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3

    def __str__(self) -> str:
        return self.name


# ======================================================================================================================
#   Class
# ======================================================================================================================
@dataclass(init=False)
class LocalExperimentTracking(ExperimentTracking):
    # ------- attributes -------------------------------------------------------
    # ------- public -------
    status: Execution = None
    platform: ComputingPlatform = None
    parameters: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)

    # ------- private -------
    _time_start: float = 0  # in epoch
    _time_end: float = None  # in epoch
    _destination_file: Path = None
    _summary_file: Path = None

    # ------- constructors -----------------------------------------------------
    def __init__(self, id: str = None,
                 is_active: bool = True,
                 destination_file: Union[Path, str] = PROJECT_ROOT / 'experiment',
                 summary_table: Union[Path, str] = PROJECT_ROOT / 'experiment' / 'summary.csv'):
        """ The creation of the object is considered as the start of the experiment """
        # initialize parent class
        super().__init__(id, is_active)

        # avoid non-required computations
        if not self.is_active:
            return

        # initialize collections
        self.parameters = {}
        self.results = {}

        # filename
        self._summary_file = Path(summary_table)
        self._destination_file = LocalExperimentTracking._craft_filename(destination_file, self.id)
        self._destination_file.parent.mkdir(parents=True, exist_ok=True)

        # platform information
        self.platform = ComputingPlatform()

        # execution status
        self.status = Execution.RUNNING
        self._time_start = time.time()

        # register __del__ callback
        atexit.register(self.destructor)

    def destructor(self):
        """ Exploiting the RAII paradigm, the experiment is logged when this object ceases to exist """
        # terminate experiment
        self.complete()  #TODO if exception, then use fail() instead

        # serialize attributes
        experiment_info: dict = self.asdict()

        # log experiment parameters
        with self._destination_file.open('w') as file:
            if self._destination_file.suffix == '.json':
                json.dump(experiment_info, file)
            elif self._destination_file.suffix in ['.yaml', '.yml']:
                yaml.dump(experiment_info, file)
            else:
                raise Exception(f'Specified format {self._destination_file.suffix} is not supported')

        # update experiment summary
        ExperimentTracking._csv_append(self._summary_file, experiment_info)

    # ------- properties -------------------------------------------------------
    @property
    def time_execution(self) -> str:
        t_last: float = self._time_end if self._time_end else time.time()  # provide time up to now
        t_elapsed: float = t_last - self._time_start  # compute duration
        return str(timedelta(seconds=t_elapsed))  # human readable

    @property
    def time_start(self) -> str:
        return str(datetime.fromtimestamp(self._time_start))

    # ------- methods ----------------------------------------------------------
    def register_parameter(self, key: str, value: Any):
        """ """
        if not self.is_active:
            return

        key, value = str(key), str(value)  # serialize input
        nested_dict: dict = reduce(lambda res, cur: {cur: res}, reversed(key.split("/")), value)  # split / as dictionary levels
        self.parameters = self.parameters | nested_dict  # merge two dictionaries
        return

    def register_result(self, key: str, value: Any):
        """ """
        if not self.is_active:
            return

        key, value = str(key), str(value)  # serialize input
        nested_dict: dict = reduce(lambda res, cur: {cur: res}, reversed(key.split("/")), value)  # split / as dictionary levels
        self.results = self.results | nested_dict  # merge two dictionaries
        return

    def already_exists(self, id: str) -> bool:
        """ """
        if not self.is_active:
            return


    def complete(self):
        self._time_end = time.time()
        self.status = Execution.SUCCESS

    def fail(self):
        self._time_end = time.time()
        self.status = Execution.FAILED

    def asdict(self) -> dict:
        def _parse(value: Any) -> str:
            if dataclasses.is_dataclass(value):  # parse dataclass
                value = asdict(value)
            # elif isinstance(value, object):  # parse object
            #     value = {k: str(v) for k, v in value.__dict__.items() if not k.startswith('_')}
            elif isinstance(value, Enum):  # parse enum
                value = str(value)
            return value
        output: dict = {k: _parse(v) for k, v in self.__dict__.items() if not k.startswith('_')}  # exclude private attributes
        return output

    # ------- internal facilities ----------------------------------------------
    @staticmethod
    def _craft_filename(filename: Union[Path, str], run_id: str) -> Path:
        filename: Path = Path(filename)
        # check if it's already a file
        if filename.suffix != '':
            if filename.suffix not in ['.json', '.yaml', '.yml']:
                raise Exception(f'ExperimentTracker is not able to handle {filename.suffix} format. Select JSON/YAML/CSV instead.')
        # otherwise is a folder, thus append experiment ID as filename
        else:
            filename = filename / (run_id + '.json')  # JSON format as default
        # if already existing, append progressive index
        index: int = 2
        pattern: str = filename.stem + '_{}' + filename.suffix
        while filename.exists():
            filename = filename.parent / pattern.format(index)
            index += 1
        # provide outcome
        return filename  # is a file

    @staticmethod
    def _csv_append(filename: Union[Path, str], row: dict):
        def _flatten(data: dict, root_name: str = "") -> dict:
            output: dict = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    res = _flatten(v, root_name=root_name + k + "/")  # recursive call
                    output.update(res)
                else:
                    output[root_name + k] = v
            return output
        df_prev: pd.DataFrame = pd.read_csv(filename) if Path(filename).exists() else pd.DataFrame()
        df_curr: pd.DataFrame = pd.DataFrame.from_dict(_flatten(row), orient='index').T  # double transpose, otherwise the dataframe is empty
        df_prev = df_prev.append(df_curr)
        df_prev.to_csv(filename, index=False)


# ======================================================================================================================
#   Entry point
# ======================================================================================================================
if __name__ == '__main__':
    """ Debug entrypoint """
    exp = LocalExperimentTracking(id='sarabanda')

    # exp.parameters["learning rate"] = 0.03
    # exp.results["accuracy"] = 0.95
    print(asdict(exp))
    r = exp.asdict()
    #ExperimentTracking._csv_append(exp._summary_file, r)
    #del exp
    pass


