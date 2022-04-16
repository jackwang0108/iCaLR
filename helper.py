# Standard Library
import platform
from typing import *
from pathlib import PosixPath, WindowsPath, Path

# Third-party Library
import numpy as np

# Select System
system = platform.uname().system

# Decide Path
Path: Union[PosixPath, WindowsPath]
if system == "Windows":
    Path = WindowsPath
elif system == "Linux":
    Path = PosixPath
else:
    raise NotImplementedError


class PathConfig:
    base: Path = Path(__file__).resolve().parent
    logs: Path = base.joinpath("logs")
    datasets: Path = base.joinpath("datasets")
    runs: Path = base.joinpath("runs")


class DatasetPath:
    class _Cifar100:
        global system
        base: Path
        if system == "Windows":
            base = PathConfig.datasets.joinpath("cifar100-windows")
        elif system == "Linux":
            base = PathConfig.datasets.joinpath("cifar100-linux")
        meta: Path = base.joinpath("meta")
        test: Path = base.joinpath("test")
        train: Path = base.joinpath("train")

    class _tinyImageNet:
        global system
        base: Path
        if system == "Windows":
            base = PathConfig.datasets.joinpath("tinyimagenet-windows")
        elif system == "Linux":
            base = PathConfig.datasets.joinpath("tinyimagenet-linux")
        train: Path = base.joinpath("train")
        val: Path = base.joinpath("val")
        test: Path = base.joinpath("test")
        wnids: Path = base.joinpath("wnids.txt")
        words: Path = base.joinpath("words.txt")

    def __init__(self):
        self.Cifar100 = self._Cifar100
        self.tinyImageNet = self._tinyImageNet

    def __str__(self):
        return "DatasetPath for iCaRL, containing Cifar100 and ImageNet2012"


class Evaluator:
    def __init__(self):
        pass

    def add_batch(self):
        pass

    def BWT(self):
        pass

    def ACC(self):
        pass

    def FWT(self):
        pass

    def Ft(self):
        pass


if __name__ == "__main__":
    # check paths
    dp = DatasetPath()
    for p in dp.Cifar100.__dict__.values():
        if isinstance(p, Path):
            print(p, p.exists())
    for p in dp.tinyImageNet.__dict__.values():
        if isinstance(p, Path):
            print(p, p.exists())
