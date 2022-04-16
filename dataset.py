# Standard Library
import pickle
from typing import *

# Third Party Libray
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

# Torch Library
import torch
import torch.utils.data as data


# My Library
from helper import DatasetPath, PathConfig, visualize, labels, num2label, label2num



class Cifar100(data.Dataset):
    __name__ = "Cifar100"

    def __init__(self, split: str, trainval_ratio: float=0.1) -> None:
        super().__init__()
        assert split in (s := ["train", "val", "test"]), f"{Fore.RED}Invalid split: {split}, please select in {s}"
        self.split: str = split

        self._image, self._label, self.class2idx = self.load(train_val_ratio=trainval_ratio)

        self.set_visible_class(labels)
    
    def __len__(self) -> int:
        return len(self.visible_label)
    
    def __getitem__(self, index: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label =  self.visible_image[index], self.visible_label[index]
        return image, label

    def set_visible_class(self, visible_class: Union[List[int], List[str]]):
        visible_class = [visible_class] if isinstance(visible_class, (int, str)) else visible_class
        _check = [type(i).__name__ for i in visible_class]
        assert len(s:=np.unique(_check)), f"all object in the visible should be same type, but you offer mixed types: {s}"
        
        # get converter
        if s[0] == "str":
            _converter = lambda cls: self.class2idx[label2num[cls]]
        else:
            _converter = lambda cls: self.class2idx[cls]

        visible_class_idx: List[np.ndarray] = []
        for cls in visible_class:
            visible_class_idx.append(_converter(cls))
        visible_class_idx = torch.concat(visible_class_idx)
        
        # set visible image and label
        self.visible_image, self.visible_label = self._image[visible_class_idx], self._label[visible_class_idx]

    
    def load(self, train_val_ratio: float)  -> Tuple[torch.Tensor, torch.Tensor, Dict[int, np.ndarray]]:
        # same as normal classification
        if self.split == "test":
            with DatasetPath.Cifar100.test.open(mode="rb") as f:
                # {b"filenames": List[bytes], 
                #  b"batch_label": bytes, 
                #  b"fine_labels": List[int], 
                #  b"coarse_labels": List[int], 
                #  b"data": np.ndarry}
                test_data: Dict[bytes, Union[bytes, List[int], List[bytes], np.ndarray]]
                test_data = pickle.load(f, encoding="bytes")
            image: torch.Tensor = torch.from_numpy(test_data[b"data"]).reshape(-1, 3, 32, 32).float() / 255
            label: torch.Tensor = torch.from_numpy(np.array(test_data[b"fine_labels"])).long()
        else:
            with DatasetPath.Cifar100.train.open(mode="rb") as f:
                # {b"filenames": List[bytes], 
                #  b"batch_label": bytes, 
                #  b"fine_labels": List[int], 
                #  b"coarse_labels": List[int], 
                #  b"data": np.ndarry}
                train_data: Dict[bytes, Any]
                train_data = pickle.load(f, encoding="bytes")
            image: torch.Tensor = torch.from_numpy(train_data[b"data"]).reshape(-1, 3, 32, 32).float() / 255
            label: torch.Tensor = torch.from_numpy(np.array(train_data[b"fine_labels"])).long()

            # decide for train and val
            if not (p:=PathConfig.base.joinpath("train_val.npz")).exists():
                # generate train and val split
                val_num = int(len(image) * train_val_ratio)
                idx = np.arange(len(image))
                np.random.shuffle(idx)
                val_idx, train_idx = idx[:val_num], idx[val_num:]
                np.savez(p, val=val_idx, train=train_idx)
            else:
                z = np.load(p)
                val_idx, train_idx = z["val"], z["train"]
            
            # get train/val data and label
            if self.split == "train":
                image, label = image[train_idx], label[train_idx]
            else:
                image, label = image[val_idx], label[val_idx]
        
        # get num2idx dict
        class2idx: Dict[int, torch.Tensor] = {i: torch.where(label == i)[0] for i in num2label.keys()}

        return image, label, class2idx

    @staticmethod
    def collate_fn(x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]):
        raise NotImplementedError


if __name__ == "__main__":
    c100 = Cifar100(split="train")
    c100.set_visible_class(labels[9])

    # check loader and image
    from PIL import Image
    import matplotlib.pyplot as plt

    loader = data.DataLoader(c100, batch_size=64, shuffle=True)
    for image, label in loader:
        a = visualize(image, label)
        a = Image.fromarray(a).show()
        break
