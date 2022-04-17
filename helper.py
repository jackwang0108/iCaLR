# Standard Library
import os
import pickle
import platform
from typing import *
from pathlib import PosixPath, WindowsPath, Path

# Third-party Library
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Select System
system = platform.uname().system


# Decide Path
# Path: Union[PosixPath, WindowsPath]
# if system == "Windows":
#     Path = WindowsPath
# elif system == "Linux":
#     Path = PosixPath
# else:
#     raise NotImplementedError


class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    logs: Path = base.joinpath("logs")
    datasets: Path = base.joinpath("datasets")
    runs: Path = base.joinpath("runs")


class DatasetPath:
    class Cifar100:
        global system
        base: Path
        if system == "Windows":
            base = ProjectPath.datasets.joinpath("cifar100-windows")
        elif system == "Linux":
            base = ProjectPath.datasets.joinpath("cifar100-linux")
        meta: Path = base.joinpath("meta")
        test: Path = base.joinpath("test")
        train: Path = base.joinpath("train")

    class tinyImageNet:
        global system
        base: Path
        if system == "Windows":
            base = ProjectPath.datasets.joinpath("tinyimagenet-windows")
        elif system == "Linux":
            base = ProjectPath.datasets.joinpath("tinyimagenet-linux")
        train: Path = base.joinpath("train")
        val: Path = base.joinpath("val")
        test: Path = base.joinpath("test")
        wnids: Path = base.joinpath("wnids.txt")
        words: Path = base.joinpath("words.txt")

    def __str__(self):
        return "DatasetPath for iCaRL, containing Cifar100 and ImageNet2012"


# load label
with DatasetPath.Cifar100.meta.open(mode="rb") as f:
    meta: Dict[str, Any] = pickle.load(f)
labels = meta["fine_label_names"]
label2num = dict(zip(labels, range(len(labels))))
num2label = dict(zip(label2num.values(), label2num.keys()))


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


def visualize(image: Union[torch.Tensor, np.ndarray],
              cls: Union[None, int, str, torch.Tensor, np.ndarray] = None) -> np.ndarray:
    """
    visualize will visualize given image(s) and return a grid of all given image(s) with label(s) (if provided)

    Args:
        image (Union[torch.Tensor, np.ndarray]): images to display, should be in the shape of [channel, width, height] or [batch, channel, width, height]
        cls (Union[None, int, str, torch.Tensor, np.ndarray], optional): label(s) of all given image(s). Defaults to None.

    Returns:
        np.ndarray: rendered iamges ([height, width, channel]), can be display by PIL or matplotlib
    """
    num = 1 if image.ndim == 3 else image.shape[0]
    cls = np.array([""] * num) if cls is None else cls
    cls = np.array([cls]) if isinstance(cls, (int, str)) else cls
    cls = cls.numpy() if isinstance(cls, torch.Tensor) else cls
    try:
        assert num == len(cls), f"{num} images with {len(cls)} labels"
    except TypeError:
        cls = np.array([cls.item()])
    image: torch.Tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    assert image.shape[
               1] == 3, f"shape of image should be [batch_size, channel, width, height] or [channel, width, height]"
    image = image.permute(0, 2, 3, 1)

    cols = int(np.sqrt(num))
    rows = num // cols + (0 if num % cols == 0 else 1)

    if isinstance(cls[0], str):
        converter = lambda x: x
    else:
        converter = lambda x: num2label[x]

    ax: List[axes.Axes]
    fig: figure.Figure
    fig, ax = plt.subplots(nrows=rows, ncols=cols, tight_layout=True, figsize=(1 * rows, 2 * cols))
    for i in range(rows):
        if rows == 1 and cols == 1:
            ax.imshow(image[i])
            ax.set_title(converter(cls[i]))
            ax.set_axis_off()
        elif cols == 1 and rows > 1:
            ax[i].imshow(image[i])
            ax[i].set_title(converter(cls[i]))
            ax[i].set_axis_off()
        else:
            for j in range(cols):
                idx = i * cols + j - 1
                if idx < num:
                    ax[i][j].imshow(image[idx])
                    ax[i][j].set_title(converter(cls[idx]))
                ax[i][j].set_axis_off()
    # plt.subplots_adjust()
    canvas = fig.canvas
    canvas.draw()

    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))


def legal_converter(path: Path):
    global system
    if system == "Windows":
        illegal_char = ["<", ">", ":", "\"", "'", "/", "\\", "|", "?", "*"]
    elif system == "Linux":
        illegal_char = ["\\"]
    relative_path: List[str] = list(str(path.relative_to(ProjectPath.base)).split("\\"))
    for idx in range(len(relative_path)):
        for cc in illegal_char:
            relative_path[idx] = relative_path[idx].replace(cc, "_")
    return ProjectPath.base.joinpath(*relative_path)


if __name__ == "__main__":
    # check paths
    # dp = DatasetPath()
    # for p in dp.Cifar100.__dict__.values():
    #     if isinstance(p, Path):
    #         print(p, p.exists())
    # for p in dp.tinyImageNet.__dict__.values():
    #     if isinstance(p, Path):
    #         print(p, p.exists())

    # check labels
    # import pprint
    #
    # pprint.pprint(labels)
    # pprint.pprint(label2num)
    # pprint.pprint(num2label)

    # test legal_converter
    import datetime
    from network import ResNet34

    lc = legal_converter(ProjectPath.runs / ResNet34.__name__ / str(datetime.datetime.now()))
    print(lc)
