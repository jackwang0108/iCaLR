# Standard Library
import pickle
from typing import *

# Third Party Library
import numpy as np
from colorama import Fore, init

init(autoreset=True)

# Torch Library
import torch
import torch.utils.data as data

# My Library
from helper import ProjectPath, DatasetPath, cifar100_labels


class ExamplarSet(data.Dataset):

    def __init__(self, k: int = 2000) -> None:
        super().__init__()

        self.k = k

        # self.examplar_set["fish"]["x"]
        # self.examplar_set["fish"]["y"]
        self.examplar_set: Dict[str, Dict[str, np.ndarray]] = {}
        self._length_list = [len(i["x"]) for i in self.examplar_set.values()]

        # self.examplar_set["fish"]["x"]
        # self.examplar_set["fish"]["y"]
        self._temp_data: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self):
        # length = sum(self._length_list)
        self._length_list = [len(i["x"]) for i in self.examplar_set.values()]
        length = sum(self._length_list)
        return length

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        ll = np.cumsum(self._length_list)
        try:
            group_idx = np.where(ll <= index)[0][-1] + 1
            in_group_shift = index - ll[group_idx - 1]
        except IndexError:
            group_idx = 0
            in_group_shift = index
        class_name = list(self.examplar_set.keys())[group_idx]
        image, label = self.examplar_set[class_name]["x"][in_group_shift], self.examplar_set[class_name]["y"][
            in_group_shift]
        return index, torch.from_numpy(image), torch.from_numpy(label)

    @torch.no_grad()
    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        labels = np.unique(y[:, 0])
        # labels = torch.unique(y[:, 0])
        for sub_label in labels:
            # idx = torch.where(y[:, 0] == sub_label)[0].tolist()
            idx = np.where(y[:, 0] == sub_label)[0].tolist()
            # if self._temp_data.get(s := cifar100_num2label[sub_label.item()], None) is not None:
            if (s := cifar100_labels.cifar100_num2label[sub_label.item()]) in self._temp_data.keys():
                self._temp_data[s]["x"] = np.concatenate(
                    [self._temp_data[s]["x"], x[idx]], axis=0
                )
                self._temp_data[s]["y"] = np.concatenate(
                    [self._temp_data[s]["y"], y[idx]], axis=0
                )
                # self._temp_data[s]["x"] = torch.cat(
                #     (self._temp_data[s]["x"], x[idx].clone().detach()), dim=0)
                # self._temp_data[s]["y"] = torch.cat(
                #     [self._temp_data[s]["y"], y[idx].clone().detach()], dim=0)
            else:
                t_dict = {}
                t_dict["x"] = x[idx]
                t_dict["y"] = y[idx]
                # t_dict["x"] = x[idx].clone().detach()
                # t_dict["y"] = y[idx].clone().detach()
                self._temp_data[s] = t_dict

    def reduce_examplar_set(self):
        self.m = self.k // len(self.examplar_set)
        for class_name in self.examplar_set.keys():
            self.examplar_set[class_name]["x"] = self.examplar_set[class_name]["x"][:self.m, :]
            self.examplar_set[class_name]["y"] = self.examplar_set[class_name]["y"][:self.m, :]

    @torch.no_grad()
    def construct_examplar_set(self, phi: 'iCaLRNet', temp: bool = False) -> Union[None, Dict[str, torch.Tensor]]:
        label_feature: Dict[str, torch.Tensor] = {}

        p = next(phi.parameters())
        device, dtype = p.device, p.dtype

        if temp:
            t = {}
        for sub_label in self._temp_data.keys():
            x = torch.from_numpy(self._temp_data[sub_label]["x"]).to(device=device, dtype=dtype)
            label_feature[sub_label] = phi.feature_extractor(x).cpu()
            # label_feature[sub_label] = phi.feature_extractor(self._temp_data[sub_label]["x"])
            # mean_feature = mu
            mean_feature = torch.mean(label_feature[sub_label], dim=0)

            # Attention: a little modification for speeding up calculation
            idx = torch.argsort(((label_feature[sub_label] - mean_feature) ** 2).sum(dim=1), descending=True)

            save_num = getattr(self, "m", self.k // len(self._temp_data))
            if save_num > (l := len(self._temp_data[sub_label]["x"])):
                repeat_time = save_num // l
                save_idx = idx.repeat(repeats=(1, repeat_time)).squeeze()
                save_idx = torch.hstack((save_idx, idx[:save_num - len(save_idx)]))
            else:
                save_idx = idx[:save_num]

            if not temp:
                self.examplar_set[sub_label] = {}
                self.examplar_set[sub_label]["x"] = self._temp_data[sub_label]["x"][save_idx]
                y = self._temp_data[sub_label]["y"][save_idx]
                y[:, 1] = cifar100_labels.cifar100_label2num[sub_label]
                self.examplar_set[sub_label]["y"] = y
            else:
                t[sub_label] = {}
                t[sub_label]["x"] = torch.from_numpy(self._temp_data[sub_label]["x"][save_idx])
                y = self._temp_data[sub_label]["y"][save_idx]
                y[:, 1] = cifar100_labels.cifar100_label2num[sub_label]
                t[sub_label]["y"] = torch.from_numpy(y)
        if temp:
            return t

        self._length_list = [len(i["x"]) for i in self.examplar_set.values()]

    def update_q(self, class_name: str, q: torch.Tensor) -> torch.Tensor:
        assert (l1 := len(self.examplar_set[class_name]["y"])) == (l2 := len(
            q)), f"{Fore.RED}Incooperate shape, q should be the shape of {l1} but you offered: {l2}"
        # Attention: the following one will keep the reference of q and original y, which will result the GPU memory overflow
        # Attention: so, do not use tensor, use numpy, and torch.Tensor.cpu will return a copy
        # self.examplar_set[class_name]["y"][:, 2:2+q.size(1)] = q
        q: np.ndarray = q.detach().cpu().numpy()
        all_length = self.examplar_set[class_name]["y"].shape[1]
        new = np.hstack(
            (self.examplar_set[class_name]["y"][:, :2],
             q,
             np.zeros(shape=(q.shape[0], all_length - 2 - q.shape[1]))
            )
        )
        # new = torch.hstack((self.examplar_set[class_name]["y"][:, :2], q.cpu(), torch.zeros(q.size(0), all_length - 2 - q.size(1))))
        self.examplar_set[class_name]["y"] = new
        return torch.from_numpy(new)

    # def update_g(self, index: torch.Tensor, value: Union[Number, torch.Tensor], dim: int):
    #     """
    #     update will update the labels in the last two dimension

    #     Args:
    #         index (torch.Tensor): index of examples
    #         value (Union[Number, torch.Tensor]): value to set
    #         dim (int): dimentsion to set
    #     """
    #     assert not dim == 0, f"{Fore.RED}You cannot changed ground truth label"
    #     if isinstance(value, torch.Tensor) and value.nelement > 1:
    #         assert (s1:=self._label[:, dim][index].shape) == (s2:=value.shape), f"{Fore.RED}Incooperate shape, trying to assign value of shape {s2} tp {s1}"
    #     # ! do not use the following, since self._label[index] will first return a copy
    #     # self._label[index][:, dim] = value
    #     self.visible_label[index, dim] = value


class Cifar100(data.Dataset):
    __name__ = "Cifar100"

    def __init__(self, split: str, examplar_set: ExamplarSet = None, trainval_ratio: float = 0.1,
                 refresh: bool = False) -> None:
        super().__init__()
        assert split in (s := ["train", "val", "test"]
                         ), f"{Fore.RED}Invalid split: {split}, please select in {s}"
        self.split: str = split

        self._image: np.ndarray
        self._label: np.ndarray
        self._visible_image: np.ndarray
        self._visible_label: np.ndarray

        self._image, self._label, self.class2idx = self._load(
            train_val_ratio=trainval_ratio, refresh=refresh)

        assert split == "train" and (
            not examplar_set is None) or split != "train", f"{Fore.RED}Training set must have examplar set"
        if split == "train":
            self.examplar_set: ExamplarSet = examplar_set

        self._new_only_flag = False

        self.current_task: Union[None, Tuple[str]] = None

        self.set_all_task()

    def __len__(self) -> int:
        assert not getattr(
            self, "visible_label",
            None) is None, f"{Fore.RED}No visible image and label, you need to call self.set_task or self.set_all_task ahead"
        if self.split == "train":
            return len(self.visible_label) + len(self.examplar_set)
        return len(self.visible_label)

    def __getitem__(self, index: Any) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        __getitem__ return images with corresponding labels

        Args:
            index (int): index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (index, image, label) pairs, image of size [1, channel, width, height], label of size [1, 3], label[1]=[gt, examplar set id, q]
        """
        if not self._new_only_flag:
            if index < (l := len(self.examplar_set)):
                return self.examplar_set.__getitem__(index)
        else:
            try:
                l = len(self.examplar_set)
            except AttributeError:
                l = 0
        index -= l
        image, label = self.visible_image[index], self.visible_label[index]
        return index + l, torch.from_numpy(image), torch.from_numpy(label)

    def set_task(self, task: Union[str, Tuple[str], None] = None):
        if task is None:
            self._set_visible_class([])
            self.current_task = None
        elif isinstance(task, str):
            self._set_visible_class([task])
            self.current_task = tuple(task)
        elif isinstance(task, tuple):
            self._set_visible_class(task)
            self.current_task = task
        else:
            assert False, f"{Fore.RED}Invalid type, task should be in [str, Tuple[str], None], but you offered {type(task)}"
        return self

    def set_all_task(self):
        global cifar100_labels
        self._set_visible_class(cifar100_labels.cifar100_labels)
        self.current_task = cifar100_labels.cifar100_labels
        return self

    def new_only(self, flag: bool = True):
        self._new_only_flag = flag
        return self

    def _set_visible_class(self, visible_class: Union[int, str, Iterable[int], Iterable[str]]) -> None:
        """
        set_visible_class will set visible images of given classes

        Args:
            visible_class (Union[int, str, List[int], List[str]]): visiable class
        """
        if len(visible_class) == 0:
            self.visible_image, self.visible_label = np.empty_like(self._image), np.empty_like(self._label)
            # self.visible_image, self.visible_label = torch.empty_like(
            #     self._image), torch.empty_like(self._label)
            return
        visible_class = [visible_class] if isinstance(
            visible_class, (int, str)) else visible_class
        _check = [type(i).__name__ for i in visible_class]
        assert len(s := np.unique(
            _check)) < 2, f"all object in the visible should be same type, but you offer mixed types: {s}"

        # get converter
        if s[0] == "str":
            def _converter(cls):
                return self.class2idx[cifar100_labels.cifar100_label2num[cls]]
        else:
            def _converter(cls):
                return self.class2idx[cls]

        visible_class_idx: List[np.ndarray] = []
        for cls in visible_class:
            visible_class_idx.append(_converter(cls))
        visible_class_idx = np.concatenate(visible_class_idx, axis=0)
        # visible_class_idx = torch.concat(visible_class_idx)

        # set visible image and label
        self.visible_image, self.visible_label = self._image[
                                                     visible_class_idx], self._label[visible_class_idx]

    def _load(self, train_val_ratio: float, refresh: bool) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        # same as normal classification
        if self.split == "test":
            with DatasetPath.Cifar100.test.open(mode="rb") as f:
                # Note: {b"filenames": List[bytes],
                # Note:  b"batch_label": bytes,
                # Note:  b"fine_labels": List[int],
                # Note:  b"coarse_labels": List[int],
                # Note:  b"data": np.ndarry}
                test_data: Dict[bytes, Union[bytes,
                                             List[int], List[bytes], np.ndarray]]
                test_data = pickle.load(f, encoding="bytes")
            image = test_data[b"data"].reshape(-1, 3, 32, 32).float() / 255
            label = np.array(test_data[b"fine_labels"])
            # image: torch.Tensor = torch.from_numpy(
            #     test_data[b"data"]).reshape(-1, 3, 32, 32).float() / 255
            # label: torch.Tensor = torch.from_numpy(
            #     np.array(test_data[b"fine_labels"]))

            # get num2idx dict
            class2idx: Dict[int, np.ndarray] = {i: np.where(
                label == i)[0] for i in cifar100_labels.cifar100_num2label.keys()}
        else:
            with DatasetPath.Cifar100.train.open(mode="rb") as f:
                # Note: {b"filenames": List[bytes],
                # Note:  b"batch_label": bytes,
                # Note:  b"fine_labels": List[int],
                # Note:  b"coarse_labels": List[int],
                # Note:  b"data": np.ndarry}
                train_data: Dict[bytes, Any]
                train_data = pickle.load(f, encoding="bytes")
            image: np.ndarray = train_data[b"data"].reshape(-1, 3, 32, 32).astype(float) / 255
            label: np.ndarray = np.array(train_data[b"fine_labels"])
            # image: torch.Tensor = torch.from_numpy(
            #     train_data[b"data"]).reshape(-1, 3, 32, 32).float() / 255
            # label: torch.Tensor = torch.from_numpy(
            #     np.array(train_data[b"fine_labels"]))

            # get num2idx dict
            class2idx: Dict[int, np.ndarray] = {i: np.where(
                label == i)[0] for i in cifar100_labels.cifar100_num2label.keys()}

            # decide for train and val
            if not (p := ProjectPath.base.joinpath("train_val.npz")).exists() or refresh:
                # generate train and val split for each class
                train_idx = {}
                val_idx = {}
                for class_num, per_class_idx in class2idx.items():
                    val_num = int(len(per_class_idx) * train_val_ratio)
                    idx = np.arange(len(per_class_idx))
                    np.random.shuffle(idx)
                    val_idx[class_num], train_idx[class_num] = idx[:val_num], idx[val_num:]
                np.savez(p, val=val_idx, train=train_idx)
            else:
                z = np.load(p, allow_pickle=True)
                val_idx, train_idx = z["val"], z["train"]

            # get train/val data and label
            if self.split == "train":
                train_idx = train_idx if isinstance(
                    train_idx, dict) else train_idx.item()
                train_idx = np.concatenate(list(train_idx.values()), axis=0)
                image, label = image[train_idx], label[train_idx]
            else:
                val_idx = val_idx if isinstance(
                    val_idx, dict) else val_idx.item()
                val_idx = np.concatenate(list(val_idx.values()), axis=0)
                image, label = image[val_idx], label[val_idx]

            # update num2idx dict
            class2idx: Dict[int, np.ndarray] = {i: np.where(
                label == i)[0] for i in cifar100_labels.cifar100_num2label.keys()}

        # import warnings
        # warnings.warn(
        #     message=f"{Fore.RED}To implement Distillation loss, y is set to [N, 2+num_class], "
        #             f"{Fore.RED}2 for y_label, examplar_flag, q."
        #             f"{Fore.RED}So, default y is float, you need to change y[0] to long before calculate CEloss, "
        #             f"{Fore.RED}You can delete this warining after debug",
        #     category=UserWarning
        # )
        ones = np.ones(shape=(label.shape[0], 1))
        q = np.zeros(shape=(ones.shape[0], 100))
        label = np.hstack((label[:, np.newaxis], ones * -1, q)).astype(float)
        return image, label, class2idx

    @staticmethod
    def collate_fn(x: Tuple[torch.Tensor], y: Tuple[torch.Tensor]):
        raise NotImplementedError


if __name__ == "__main__":
    e = ExamplarSet()
    from network import iCaLRNet
    import torch.optim as optim

    net = iCaLRNet(num_class=100, target_dataset="cifar100", examplar_set=e).double()
    optimizer = optim.Adam(net.parameters())

    c100 = Cifar100(split="train", examplar_set=e)
    loader = data.DataLoader(dataset=c100, batch_size=64, shuffle=True)
    task_list = [("apple", "aquarium_fish"),
                 ("baby", "bear"), ("beaver", "bed")]
    for learn_time, task in enumerate(task_list):
        x: torch.Tensor
        y: torch.Tensor

        net.set_task(task=task)
        c100.set_task(task=task)

        # set g
        if learn_time >= 1:
            for seen_class in c100.examplar_set.examplar_set.keys():
                x = c100.examplar_set.examplar_set[seen_class]["x"]
                y = c100.examplar_set.examplar_set[seen_class]["y"]
                g = net(x)[:, :len(net.seen_classes) - len(task)]
                # == test only
                # g = torch.randn(y.size(0), 100)
                c100.examplar_set.update_q(class_name=seen_class, q=g)

        for epoch in range(70):
            # train
            net.train()
            for label, x, y in loader:
                print(x.dtype)
                print(y.dtype)
                y_pred = net(x)
                loss = net.loss_func(y_pred=y_pred, y=y)
                optimizer.step()
                print(loss)
                c100.examplar_set.add_batch(x=x, y=y)
        c100.examplar_set.construct_examplar_set(net)

# if __name__ == "__main__":
#     e = ExamplarSet()
#     c100 = Cifar100(split="train", examplar_set=e)
#     visualize(c100.visible_image[:100], c100.visible_label[:100])
