# Standard Library
import os
import pickle
import platform
from typing import *
from pathlib import PosixPath, WindowsPath, Path

# Third-party Library
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from colorama import Fore, init
from terminaltables import SingleTable
from matplotlib import figure, axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

init(autoreset=True)

# Select System
system: str = platform.uname().system


# Decide Path
Path: Union[PosixPath, WindowsPath]
if system == "Windows":
    Path = WindowsPath
elif system == "Linux":
    Path = PosixPath
else:
    raise NotImplementedError


class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    logs: Path = base.joinpath("logs")
    datasets: Path = base.joinpath("datasets")
    runs: Path = base.joinpath("runs")
    checkpoints: Path = base.joinpath("checkpoints")


for attr in ProjectPath.__dict__.values():
    if isinstance(attr, Path):
        attr.mkdir(parents=True, exist_ok=True)


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
class CifarTaskSetter:

    def __init__(self) -> None:
        import copy
        with DatasetPath.Cifar100.meta.open(mode="rb") as f:
            meta: Dict[str, Any] = pickle.load(f)
        self.all_class = meta["fine_label_names"]
        self.origin_class = copy.deepcopy(meta["fine_label_names"])
        self._gen_converter()
    
    def _gen_converter(self):
        self.label2num = dict(zip(self.all_class, range(len(self.all_class))))
        self.num2label = dict(enumerate(self.all_class))
        _old_label2num = dict(zip(self.origin_class, range(len(self.origin_class))))
        self.old2new = {_old_label2num[name]:self.label2num[name] for name in self.origin_class}
        return self

    def gen_task_list(self, num_task: int = 10) -> List[Tuple[str]]:
        task_list = []
        for i in range(0, len(self.all_class), num_task):
            task_list.append(tuple(self.all_class[i: i+num_task]))
        self.task_list = task_list
        self._gen_converter()
        return task_list
    
    def set_task_list(self, task_list: List[Tuple[str]]):
        given_task = []
        for task in task_list:
            given_task.extend(task)
        self.all_class = given_task
        self.task_list = task_list
        self._gen_converter()
        return self.task_list

    def shuffle(self):
        import random
        random.shuffle(self.all_class)
        self._gen_converter()
        return self
    
    def get_name(self, class_num: int) -> str:
        return self.num2label[class_num]

    def get_num(self, class_name: str) -> int:
        return self.label2num[class_name]

cifar_task_setter = CifarTaskSetter()

# ????????????????????????confusion matrix???????????????: https://zhuanlan.zhihu.com/p/147663370
# ??????????????????, ???????????????
class ClassificationEvaluator:
    def __init__(self, num_class: int) -> None:
        self._num_class = num_class
        # rows: ground truth class, cols: predicted class
        self.top1_confusion_matrix = np.zeros(shape=(num_class, num_class))
        self.top5_confusion_matrix = np.zeros(shape=(num_class, num_class))

        # register measurement
        self._measure: Dict[str, Callable] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }

    def add_batch_top1(self, y_pred: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], seen_class:int=0):
        """
        batch_confusion_matrix is used to compute the confusion matrix of a batch
        Args:
            labels: Prediction labels of points with shape of [batch]
            num_classes: Ground truth labels of points with shape of [batch] or [batch, ]
        Returns:
            confusion matrix, represented by a tensor of [batch, num_classes, num_classes]
        """
        # check type
        assert isinstance(y_pred, (torch.Tensor, np.ndarray)), f"Not suppported type for pred_y: {type(y_pred)}"
        assert isinstance(y, (torch.Tensor, np.ndarray)), f"Not suppported type for y: {type(y)}"
        # check length
        assert (a:=len(y_pred)) == (b:=len(y)), f"None-equal predictions and ground truth, given prediction of {a} examples, but only with {b} ground truth"
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().to(device="cpu").numpy()
        y = y if isinstance(y, np.ndarray) else y.detach().to(device="cpu").numpy()

        # construc batch confusion matrix and add to self.confusion_matrix
        k_y = (y >= seen_class) & (y < self._num_class + seen_class)
        k_y_pred = (y_pred >= seen_class) & (y_pred < self._num_class + seen_class)
        k = k_y & k_y_pred

        # convert [batch, num_class] prediction scores to [batch] prediction results
        y_pred_cls = (y_pred if y_pred.ndim == 1 else y_pred.argmax(axis=1)).squeeze()

        confusion_matrix: np.ndarray
        # bincount for fast classification confusion matrix
        confusion_matrix = np.bincount(
            self._num_class * y[k].astype(int) + y_pred_cls[k].astype(int),
            minlength=self._num_class ** 2
        ).reshape(self._num_class, self._num_class)
        self.top1_confusion_matrix += confusion_matrix
        return confusion_matrix

    def add_batch_top5(self, y_pred: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], seen_class: int=0):
        """
        batch_confusion_matrix is used to compute the confusion matrix of a batch
        Args:
            labels: Prediction labels of points with shape of [batch]
            num_classes: Ground truth labels of points with shape of [batch] or [batch, ]
        Returns:
            confusion matrix, represented by a tensor of [batch, num_classes, num_classes]
        """
        # check type
        assert isinstance(y_pred, (torch.Tensor, np.ndarray)), f"Not suppported type for pred_y: {type(y_pred)}"
        assert isinstance(y, (torch.Tensor, np.ndarray)), f"Not suppported type for y: {type(y)}"
        # check length
        assert (a:=len(y_pred)) == (b:=len(y)), f"None-equal predictions and ground truth, given prediction of {a} examples, but only with {b} ground truth"
        # check input
        assert y_pred.ndim == 2, f"For top5 evaluation, you should input [batch, class_score] tensor or ndarray, but you offered: {y_pred.shape}"
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().to(device="cpu").numpy()
        y = y if isinstance(y, np.ndarray) else y.detach().to(device="cpu").numpy()

        # construc batch confusion matrix and add to self.confusion_matrix
        k_y = (y >= seen_class) & (y < self._num_class + seen_class)
        k_y_pred = (y_pred >= seen_class) & (y_pred < self._num_class + seen_class)
        if k_y_pred.ndim == 2:
            k = (k_y_pred & k_y[:, np.newaxis]).all(axis=1)
        else:
            k = k_y_pred & k_y

        # this could be done by torch.Tensor.topk, but for numpy, argsort is O(NlongN), following is a O(N) implementation
        # [1st, 2st, ..., 5st]
        if y_pred.shape[1] > 5:
            y_pred_cls = np.argpartition(y_pred, kth=-5, axis=1)[:, -5:][:, ::-1]
        elif y_pred.shape[1] == 5:
            y_pred_cls = y_pred
        else:
            assert False, f"{Fore.RED}Error prediction, mimum 5 precition each example, but you offered {y_pred.shape}"

        correct_mask = (y[:, np.newaxis] == y_pred_cls).any(axis=1)
        pred_yy = np.zeros_like(y)
        pred_yy[correct_mask] = y[correct_mask]
        pred_yy[~correct_mask] = y_pred_cls[~correct_mask, 0]

        confusion_matrix: np.ndarray
        # bincount for fast classification confusion matrix
        confusion_matrix = np.bincount(
            self._num_class * y[k].astype(int) + pred_yy[k].astype(int),
            minlength=self._num_class ** 2
        ).reshape(self._num_class, self._num_class)
        self.top5_confusion_matrix += confusion_matrix
        return confusion_matrix
    
    def accuracy(self, top: int = 5) -> np.float64:

        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray =  self.__getattribute__(f"top{top}_confusion_matrix")
        with np.errstate(divide='ignore', invalid='ignore'):
            acc: np.ndarray = confusion_matrix.trace() / confusion_matrix.sum()
        return np.nan_to_num(acc)

    def precision(self, top: int = 5) -> Tuple[np.ndarray, np.float64]:

        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray =  self.__getattribute__(f"top{top}_confusion_matrix")

        # ignore zero division error, invalid division warning
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision: np.ndarray = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
            per_class_precision[np.isnan(per_class_precision)] = 0
        mean_precision = per_class_precision.mean()
        return per_class_precision, mean_precision

    def recall(self, top: int = 5):
        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray =  self.__getattribute__(f"top{top}_confusion_matrix")

        # ignore zero division error, invalid division warning
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall: np.ndarray = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
            per_class_recall[np.isnan(per_class_recall)] = 0
        mean_recall = per_class_recall.mean()
        return per_class_recall, mean_recall
    
    def new_epoch(self):
        self.top1_confusion_matrix = np.zeros(shape=(self._num_class, self._num_class))
        self.top5_confusion_matrix = np.zeros(shape=(self._num_class, self._num_class))
    
    def make_grid(self, title: str, labels: List[str], top: int = 5):
        assert len(labels) == self._num_class, f"Evaluator is initialized with {self._num_class} classes, but reveive only {len(labels)} labels."
        data = []
        index = []
        column = labels + ["Mean"]
        last_col = []
        for measure_name, meansure in self._measure.items():
            index.append(measure_name)
            result = meansure(top=top)
            if isinstance(result, tuple):
                a = np.random.randn(10, 10)
                np.round(a,)
                data.append(np.round(result[0], decimals=4))
                last_col.append(np.round(result[1], decimals=4))
            else:
                data.append(np.array(["---"] * (len(column) - 1)))
                last_col.append(np.round(result, decimals=4))
        data = np.array(data)
        last_col = np.array(last_col)[:, np.newaxis]
        df = pd.DataFrame(np.hstack((data, last_col)), index=index, columns=column).T
        data = df.to_numpy()
        index = df.index.to_numpy()
        column = df.columns.tolist()
        data = np.hstack((index[:, np.newaxis], data)).tolist()
        column.insert(0, "class")
        data.insert(0, column)
        data[-1] = [f"{Fore.BLUE}{i}{Fore.RESET}" for i in data[-1]]
        table = SingleTable(data)
        table.title = title + f" top {top}"
        return table.table



class ContinualLearningEvaluator:
    def __init__(self, num_task: int):
        self._num_task = num_task

        self.top1_R = np.zeros(shape=(num_task, num_task))
        self.top5_R = np.zeros(shape=(num_task, num_task))

        self.overall_task: List[Tuple[str]] = []

    def after_new_task(self, task: Union[str, Tuple[int]]):
        if isinstance(task, str):
            self.overall_task.append(tuple(task))
        elif isinstance(task, tuple):
            self.overall_task.append(task)
        else:
            assert False, f"{Fore.RED}Invalid type: {type(task)}"
        self.top1_R = np.zeros(shape=(self._num_task, self._num_task))
        self.top5_R = np.zeros(shape=(self._num_task, self._num_task))

    def start_test_task(self, task: Union[str, Tuple[int, str]], test_task_idx: int):
        self._top1_classification_evaluator = ClassificationEvaluator(num_class=len(task))
        self._top5_classification_evaluator = ClassificationEvaluator(num_class=len(task))
        self._test_task_idx = test_task_idx
    
    def end_test_task(self):
        self.top1_R[len(self.overall_task)-1, self._test_task_idx] = self._top1_classification_evaluator.accuracy(top=1)
        if self.if_top5:
            self.top5_R[len(self.overall_task)-1, self._test_task_idx] = self._top5_classification_evaluator.accuracy(top=5)
        else:
            return 0
    
    def add_batch(self, y_pred: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]):
        self.if_top5 = True if y_pred.shape[1] >= 5 else False
        # check type
        assert isinstance(y_pred, (torch.Tensor, np.ndarray)), f"{Fore.RED}Not suppported type for pred_y: {type(y_pred)}"
        assert isinstance(y, (torch.Tensor, np.ndarray)), f"Not suppported type for y: {type(y)}"
        # check length
        assert (a:=len(y_pred)) == (b:=len(y)), f"None-equal predictions and ground truth, given prediction of {a} examples, but only with {b} ground truth"
        # check input
        assert y_pred.ndim == 2, f"For both top1 and top5 evaluation, you should input [batch, pred_cls(5)] tensor or ndarray, but you offered: {y_pred.shape}"
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().to(device="cpu").numpy()
        y = y if isinstance(y, np.ndarray) else y.detach().to(device="cpu").numpy()

        top1_pred = y_pred[:, 0]
        top5_pred = y_pred[:, :]

        self._top1_classification_evaluator.add_batch_top1(y_pred=top1_pred, y=y)
        if top5_pred.shape[1] >= 5:
            self._top5_classification_evaluator.add_batch_top5(y_pred=top5_pred, y=y)


    def BWT(self, top: int = 1):
        t = len(self.overall_task)
        r: np.ndarray = getattr(self, f"top{top}_R")[:t, :t]
        return (r[-1, :] - r.diagonal()).mean()

    def mACC(self, top: int = 1):
        t = len(self.overall_task)
        r: np.ndarray = getattr(self, f"top{top}_R")[:t, :t]
        return r[-1, :].mean()

    def FWT(self, top: int = 1):
        raise NotImplementedError

    def Ft(self, top: int = 1):
        raise NotImplementedError


def visualize(image: Union[torch.Tensor, np.ndarray],
              cls: Union[None, int, str, torch.Tensor, np.ndarray] = None,
              pil: bool = False) -> Union[np.ndarray, Image.Image]:
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
    cls: np.ndarray = cls.detach().cpu().numpy() if isinstance(cls, torch.Tensor) else cls
    cls = cls[:, 0] if cls.ndim == 2 and cls.shape[1] >= 2 else cls
    try:
        assert num == len(cls), f"{num} images with {len(cls)} labels"
    except TypeError:
        cls = np.array([cls.item()])
    image: torch.Tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.from_numpy(image)

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
        converter = lambda x: cifar_task_setter.get_name(x)

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
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    image = np.fromstring(s, dtype=np.uint8).reshape((height, width, 4))
    if pil:
        return Image.fromarray(image)
    else:
        return image


def legal_converter(path: Path) -> Path:
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
    # import datetime
    # from network import ResNet34

    # lc = legal_converter(ProjectPath.runs / ResNet34.__name__ / str(datetime.datetime.now()))
    # print(lc)

    # test evaluator
    # ce = ClassificationEvaluator(num_class=10)
    # y = np.repeat(np.arange(0, 10)[np.newaxis, :], repeats=10).flatten()
    # pred_y = np.zeros(shape=(100))
    # # top1
    # # ce.add_batch(pred_y=pred_y, y=y)
    # # ce.add_batch_top1(y_pred=y, y=y)
    # # top5
    # pred_y = np.random.random(size=(100, 10))
    # ce.add_batch_top1(y_pred=pred_y, y=y)
    # ce.add_batch_top5(y_pred=pred_y, y=y)
    #
    # print(ce.accuracy(top=1))
    # print(ce.accuracy(top=5))
    # print(ce.precision(top=1))

    # print(ce.make_grid(title="Epoch 1", top=1, labels=cifar100_labels[:10]))
    pass