# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
from dataclasses import dataclass, field

import psutil
import platform
import cpuinfo
import GPUtil


# ======================================================================================================================
#   Class
# ======================================================================================================================
@dataclass(init=True)
class ComputingPlatform:
    # ------- attributes -------------------------------------------------------
    hostname: str = ""
    operating_system: str = ""
    language: str = ""
    ram: {str, str} = field(default_factory=dict)
    cpu: {str, str} = field(default_factory=dict)
    gpu: [dict] = field(default_factory=list)

    # ------- constructors -----------------------------------------------------
    def __post_init__(self):
        """ Method invoked by dataclass as last call in __init__() """
        # ------- hostname and OS -------
        uname: platform.uname_result = platform.uname()
        self.hostname = uname.node  # name of the current PC
        self.operating_system = f"{uname.system} {uname.release}"  # version of the OS (Windows/Linux)

        # ------- CPU and language -------
        cpu_info: {str, str} = cpuinfo.get_cpu_info()
        self.language = f"Python {cpu_info['python_version']}"  # python version
        self.cpu['architecture'] = f"{cpu_info['arch']} ({cpu_info['arch_string_raw']})"  # architecture x86 vs. x64
        self.cpu['version'] = cpu_info['brand_raw']  # i3/i5/i7 and generation
        self.cpu['manufacturer'] = cpu_info['vendor_id_raw']  # who produced the CPU

        # ------- memory information -------
        svmem: psutil._common.sswap = psutil.virtual_memory()
        self.ram['total'] = ComputingPlatform.get_size(svmem.total)
        swap: psutil._common.sswap = psutil.swap_memory()
        self.ram['swap'] = ComputingPlatform.get_size(swap.total)

        # ------- GPU list -------
        gpus: list = GPUtil.getGPUs()  # retrieve info of all the GPUs in the machine
        for gpu in gpus:
            gpu_info: dict = {
                'name': gpu.name,  # name of the current GPU
                'memory': f"{gpu.memoryTotal}MB"  # total memory of the current GPU
            }
            self.gpu.append(gpu_info)

    # ------- methods ----------------------------------------------------------
    @staticmethod
    def get_size(bytes: float, suffix: str = "B") -> str:
        """
        Scale bytes to its proper format
        e.g:
            1253656 => '1.20MB'
            1253656678 => '1.17GB'
        """
        factor: float = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor
