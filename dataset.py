# Standard Library
import pickle
from typing import *

# Third Party Libray
from colorama import Fore, Style, init
init(autoreset=True)

# Torch Library
import torch
import torch.utils.data as data


# My Library
from helper import DatasetPath



class Cifar100(data.Dataset):
    def __init__(self, split: str) -> None:
        super().__init__()
        assert split in (s := ["train", "val", "test"]), f"{Fore.RED}Invalid split: {split}, please select in {s}"
        self.split: str = split
        