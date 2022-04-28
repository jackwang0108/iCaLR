# Torch Library
import torch
import torch.nn as nn

# Standard Library
from typing import *

# Third-party Libraries
from colorama import Fore, init

init(autoreset=True)

# My Library
from dataset import ExamplarSet


# Since the original iCaLR is implemented using lasagne and Theano, I really cannot read the original codes
# the architecture of the ResNet used in my pytorch re-implementation is ResNet original paper 

class _BasicResidualBlock(nn.Module):
    __name__ = "ResidualBlock"

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 with_projection: Union[bool, None] = None) -> None:
        """
        Pytorch implementation of basic residual block for shallow resnet, e.g., resnet18 and resnet34
        Args:
            in_channels (int): in channels
            out_channels (int): out channels
            stride (int): stride of the first 3*3 convolution layer, resnet do not use maxpool, instead, it uses
                convolution as downsample. Set stride to 1 will keep the input shape while set to 2 means downsample
                the image to half
            with_projection (Union[bool, None]): if use project in the shortcut connection, set to None will let the
                model decide.
        """
        super(_BasicResidualBlock, self).__init__()
        # Residual path, bias is omitted (as said in original paper)
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=(3, 3),
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

        # shortcut connection
        # with 1D convolution (?)
        if with_projection is None:
            self.with_projection = True if in_channels != out_channels else False
        else:
            self.with_projection = with_projection

        if with_projection or stride == 2:
            # using 2D convolution of 1*1 kernel to match the input and output channels
            # or downsample
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # output relu
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the original notation in paper
        fx = self.residual_path(x)
        x = self.shortcut(x)
        hx = fx + x
        return self.output_relu(hx)


class _BottleneckResidualBlock(nn.Module):
    __name__ = "ResidualBlock"

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 with_projection: Union[bool, None] = None) -> None:
        """
        Pytorch implementation of bottleneck residual block for shallow resnet, e.g., resnet18 and resnet34
        Args:
            in_channels (int): in channels
            out_channels (int): out channels
            stride (int): stride of the first 3*3 convolution layer, resnet do not use maxpool, instead, it uses
                convolution as downsample. Set stride to 1 will keep the input shape while set to 2 means downsample
                the image to half
            with_projection (Union[bool, None]): if use project in the shortcut connection, set to None will let the
                model decide.
        """
        super(_BottleneckResidualBlock, self).__init__()
        # Residual path, bias is omitted (as said in original paper)
        # using 2D convolution of 1*1 kernel to reduce parameters when process large input, ImageNet for example.
        self.residual_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut connection
        # with 1D convolution (?)
        if with_projection is None:
            self.with_projection = True if in_channels != out_channels else False
        else:
            self.with_projection = with_projection

        if with_projection or stride == 2:
            # using 2D convolution of 1*1 kernel to match the input and output shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # output relu
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the original notation in paper
        fx = self.residual_path(x)
        x = self.shortcut(x)
        hx = fx + x
        return self.output_relu(hx)


class _ResNetBase(nn.Module):
    """ResNet pytorch implementation, offering ResNet variants implementation via __init__ parameters"""
    __name__ = "ResNetBase"

    # cifar like dataset, input images are small
    cifar_like_dataset: List[str] = ["Cifar100", "cifar100"]
    # ImageNet like dataset, input images are large
    imagenet_like_datasets: List[str] = ["tinyImageNet200"]

    def __init__(self, target_dataset: str, num_blocks: List[int], num_class: Union[int, None] = None):
        if not target_dataset in (a := self.cifar_like_dataset + self.imagenet_like_datasets):
            raise NotImplementedError(f"Not implemented for dataset: {target_dataset}, currently available datasets are"
                                      f" {a}")
        
        super(_ResNetBase, self).__init__()

        self.target_dataset: str = target_dataset

        # first transformation layers for different datasets
        # if you want to run on you onw dataset, you need to write the transformation layers
        if self.target_dataset in self.imagenet_like_datasets:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=2),
                nn.MaxPool2d(stride=2)
            )
        elif self.target_dataset in self.cifar_like_dataset:
            self.transform = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )

        # decide block type
        block_type = _BasicResidualBlock if target_dataset in self.cifar_like_dataset else _BottleneckResidualBlock

        self.stage1 = self._make_stage(
            block_type=block_type,
            in_channels=64,
            out_channels=64,
            first_stride=1,
            num_blocks=num_blocks[0]
        )
        self.stage2 = self._make_stage(
            block_type=block_type,
            in_channels=64,
            out_channels=128,
            first_stride=2,
            num_blocks=num_blocks[1]
        )
        self.stage3 = self._make_stage(
            block_type=block_type,
            in_channels=128,
            out_channels=256,
            first_stride=2,
            num_blocks=num_blocks[2]
        )
        self.stage4 = self._make_stage(
            block_type=block_type,
            in_channels=256,
            out_channels=512,
            first_stride=2,
            num_blocks=num_blocks[3]
        )
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_class: bool = num_class
        if num_class is not None:
            self.fc = nn.Linear(in_features=512, out_features=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.transform(x)

        x1 = self.stage1(input)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        pooled: torch.Tensor = self.average_pool(x4)


        feature_map = pooled.view(pooled.shape[0], -1)
        if self.num_class is not None:
            # [batch_size, in_features]
            return self.fc(feature_map)
        return feature_map

    def _make_stage(self, block_type: Union[_BasicResidualBlock, _BottleneckResidualBlock],
                    in_channels: int, out_channels: int, num_blocks: int, first_stride: int) -> nn.Sequential:
        """
        _make_stage will make a stage. A stage means several basic/bottleneck residual blocks, in original paper, there
            are 4 stages.
        Args:
            block_type (Union[BasicResidualBlock, BottleneckResidualBlock]): type of the block
            in_channels (int): channel of input image of the stage
            out_channels (int): channel of output image of the stage
            num_blocks (int): number of blocks in this stage
            first_stride (int): stride of the the first block in the stage.
        Returns:
            nn.Sequential: stage of the layer
        """
        block_strides: List[int] = [first_stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_idx, stride in enumerate(block_strides):
            blocks.append(
                _BasicResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
            )
            if block_idx == 0:
                # only the first block will do downsample and expand channels
                in_channels = out_channels
        return nn.Sequential(*blocks)


class _ResNet34(_ResNetBase):
    __name__ = "ResNet34"

    def __init__(self, num_class: int, target_dataset: str):
        assert isinstance(num_class, int), f"{Fore.RED}Classification network must specify predicted class nums"
        super(_ResNet34, self).__init__(
            num_class=num_class,
            target_dataset=target_dataset,
            num_blocks=[3, 4, 6, 3],
        )

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor = self(x)
        result = preds.argmax(dim=1)
        return result


class DistillationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        nn.BCELoss()
        self._bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y_pred: [batch_size, len(seen_class)]
        # y: [batch_size, seen_class]
        q = y
        g = y_pred
        return self._bce_loss(y_pred, y)




class iCaLRNet(nn.Module):
    __name__ = "iCaLRNet"
    __fe__ = "ResNet"

    def __init__(self, num_class: Union[None, str], target_dataset: str, examplar_set: ExamplarSet) -> None:
        """
        __init__ iCaLRNet implementation

        Args:
            num_class (int): all class
            target_dataset (str): target dataset
        
        Notes:
            In iCaLR paper, the network make prediction via Nearest-Mean-of-Exemplars.
            But to train the network, the paper uses weight vectors.
            In detail, phi(x) denote feature extractor in the paper. And in the code, phi is a ResNet without prediction layer (last linear layer).
            The output shape of phi(x) is [batch_size, 512]. Let x denote the input, phi(x) := [d_1, d_2, ..., d_512].
            Let w^y denote weight vector, so w = [w_1, w_2, ..., w_512]^y.
            So, a_y(x) = w^y @ phi(x), and g_y(x) = 1 / (1 + exp(-a_y(x))).
            For all y in {0, 1, ..., t}
            vector a
            For a batch of 9 examples, the output is 
                x_1:    [d_1, d_2, ..., d_512]^1
                x_2:    [d_1, d_2, ..., d_512]^2
                ...     [..., ..., ..., ...  ]
                x_9:    [d_1, d_2, ..., d_512]^9
            
        """
        super().__init__()

        self.num_class = num_class
        self.target_dataset = target_dataset
        self.feature_extractor = _ResNetBase(target_dataset=target_dataset, num_blocks=[3, 4, 6, 3], num_class=None)
        if num_class is None:
            self.weights = nn.Linear(in_features=512,out_features=2, bias=False)
        else:
            self.weights = nn.Linear(in_features=512, out_features=num_class, bias=False)
        self.examplar_set = examplar_set

        self.seen_classes: List[str] = []
        self._sigmoid = nn.Sigmoid()

        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._distillation_loss = DistillationLoss()
    

    def forward(self, x: torch.Tensor, feature: bool = True) -> torch.Tensor:
        feature_vectores = self.feature_extractor(x)
        if self.training or not feature:
            train_only = self.weights(feature_vectores)
            g = self._sigmoid(train_only[:, :len(self.seen_classes)])
            return g
        else:
            return feature_vectores
    
    def add_task(self, task: Union[str, List[str]]):
        assert isinstance(task, tuple), f"{Fore.RED}Only tuple is accepted, please offer tasks like (chair, book)."
        # extend wight and copu weights
        current_device = self.weights.weight.device
        current_dtype = self.weights.weight.dtype
        origin_shape = self.weights.weight.shape
        new_weight = nn.Linear(in_features=512, out_features=len(self.seen_classes) + len(task), bias=False).to(device=current_device, dtype=current_dtype)
        new_weight.weight.data[:origin_shape[0], :] = self.weights.weight.data.detach()
        del self.weights
        self.weights = new_weight

        # update task list
        self.current_task = task
        self.seen_classes.extend(task)
        return self
    
    def loss_func(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert y.size(1) >= 2 , f"Ground truth should be [batch_size, 2+num_class], but the groud truth you give is {y.shape}"
        # get new example idx
        new_idx = torch.where(y[:, 1] == -1)[0]
        # get old example idx
        old_idx = torch.where(y[:, 1] != -1)[0]

        new_ce_loss = self._cross_entropy_loss(y_pred[new_idx], y[new_idx, 0].long())
        if len(old_idx) > 0:
            old_num = len(self.seen_classes) - len(self.current_task)
            old_distill_loss = self._distillation_loss(y_pred[old_idx, :old_num], y[old_idx, 2:2+old_num])
            return new_ce_loss, old_distill_loss
        return new_ce_loss


    @torch.no_grad()
    def inference(self, x: torch.Tensor, temp_examplar_set: Dict[str, Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        mu_all = []
        if temp_examplar_set is None:
            examplar_set = self.examplar_set.examplar_set
        else:
            examplar_set = temp_examplar_set

        for class_name in examplar_set.keys():
            x_examplar = examplar_set[class_name]["x"]
            x_examplar = x_examplar if isinstance(x_examplar, torch.Tensor) else torch.from_numpy(x_examplar)
            if x.device == torch.device("cuda:0"):
                x_examplar = x_examplar.cuda()
            mu_y = self.feature_extractor(x_examplar).mean(dim=0)
            mu_all.append(mu_y)
        mu_all: torch.Tensor = torch.vstack(mu_all)

        result = []
        for xx in self.feature_extractor(x):
            distance = torch.sqrt(((xx - mu_all)**2).sum(dim=1))
            result.append(torch.argsort(distance))
        return torch.vstack(result)


if __name__ == "__main__":
    fc = nn.Linear(in_features=512, out_features=10, bias=False)
    print(fc.weight.shape)
