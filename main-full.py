# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# standard libraries
import datetime
from pathlib import Path

# third-party libraries
import numpy as np
from colorama import Fore, init

# my libraries
from network import ResNet34
from dataset import Cifar100
from helper import ProjectPath, DatasetPath, visualize, legal_converter, labels, num2label, label2num


class FullTrainer:
    start_time = datetime.datetime.now()
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_dtype = torch.float

    def __init__(self, network: nn.Module):
        self.network = network

        self.train_loader


# todo: 写完训练
# todo: 写完dataset
# todo: 写完helper里的Evaluator
# todo: 最好可以开始训练
# todo: 注意optim的parameter要更新
# todo: 注意网络里的分类头的更新
# todo: 注意网络里要有一个predict/inference方法
# todo: 训练代码中训练顺序的问题
# todo: 首先要确保ResNet联合训练性能可以
# todo: dataset的train val划分的不合理, 参考原文实现是每一类都会划分出来几个

if __name__ == "__main__":
    ft = FullTrainer()
