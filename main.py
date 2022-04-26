"""
Notes:
    心得:
        1. 只有必要的时候在用Tensor, 因为Tensor具有 grad, view, device等复杂属性, 一般的视图也会被记录grad, 留下reference,
            导致不必要的内存开销, 所以如无必要勿用Tensor, Numpy解决就行了
        2. 如果必须要用Tensor, 或者需要使用非训练中的网络来进行计算, 一定要使用torch.no_grad
        3. 数据集中尽量使用np.ndarray, 然后可以用上torchvision.transforms
        4. 写完了之后, 多用用logging, 便于保留每一次训练结果
"""

# Torch Library
import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# Standard Library
import copy
import logging
import datetime
from typing import *
from pathlib import Path


# Third-party Library
from colorama import Fore, init

init(autoreset=True)

# My Library
from network import iCaLRNet
from dataset import Cifar100, ExamplarSet
from helper import ProjectPath, legal_converter, cifar100_labels, system
from helper import ClassificationEvaluator


class CLTrainer:
    start_time: str = str(datetime.datetime.now())
    available_device: torch.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_worker: int = 0 if system == "Windows" else 2

    default_dtype: torch.dtype = torch.float32

    def __init__(self, network: iCaLRNet, task_list: List[Tuple[str]], dry_run: bool=True, log: bool = False) -> None:
        self.train_list = task_list
        self.net = network.to(device=self.available_device, non_blocking=True)
        self.log = log
        self.dry_run: bool = dry_run

        self.optim = optim.Adam(self.net.parameters())

        # summary writer
        suffix = self.net.__class__.__name__ + f"/num_task_{len(self.train_list[0])}"
        if not self.dry_run:
            writer_path: Path = legal_converter(ProjectPath.runs / suffix /self.start_time)
            self.writer = SummaryWriter(log_dir=writer_path)
        
        # datasets
        self.train_set = Cifar100(split="train", examplar_set=self.net.examplar_set)
        self.val_set = Cifar100(split="val").new_only(flag=True)

        assert (a:=network.target_dataset) == (b:=self.train_set.__name__), f"{Fore.RED}In-cooperate network and dataset, network is for {a}, but dataset is {b}"

        # checkpoint path
        self.checkpoint_path = legal_converter(ProjectPath.checkpoints / suffix / self.start_time)
        if not dry_run:
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # log
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter(fmt="%(message)s", datefmt="")

        terminal = logging.StreamHandler()
        terminal.setLevel(logging.INFO)
        terminal.setFormatter(fmt=fmt)
        self.logger.addHandler(terminal)

        if log:
            self.log_path = legal_converter(ProjectPath.logs / suffix / self.start_time / "trian.log")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"{Fore.GREEN}Save log to {self.log_path}")
            file = logging.FileHandler(filename=str(self.log_path))
            file.setLevel(logging.INFO)
            file.setFormatter(fmt=fmt)
            self.logger.addHandler(file)


    def __del__(self):
        if getattr(self, "writer", None) is not None:
            self.writer.close()
        
        self.logger.info(f"Shutdown at {datetime.datetime.now()}, waiting...")
        # shutdown logging and flush buffer
        import time
        # wait for write to file
        time.sleep(3)
        logging.shutdown()
        
        # clean logs
        import re
        with self.log_path.open(mode="r+") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = re.sub(r"..\d\dm", "", lines[i])
            f.seek(0)
            f.write("".join(lines[:-1]))

    def train(self, n_epcoh: int = 70, early_stop: int = 20, bsize: int = 128):
        task_list = self.train_list

        self.logger.info(f"n_epoch: {n_epcoh}")
        self.logger.info(f"early_stop: {early_stop}")
        self.logger.info(f"task_list: {task_list}")

        # writer setup
        loss_step: int = 0
        if not self.dry_run:
            # save optim, feature_extractor
            self.writer.add_text(tag="Train Setup", global_step=0,
                text_string=f"Optim: {self.optim.__class__.__name__}, " + f"Feature Extractor: {self.net.__fe__}"
            )
            # save hparam
            self.writer.add_text(tag= "Train Hparam", global_step=0,
                text_string= f"lr: {self.optim.param_groups[0]['lr']}, " + f"bsize: {bsize}, " + f"epoch: {n_epcoh}, " + f"early_stop: {early_stop}"
            )
            # save task list
            for task_seq, task in enumerate(task_list):
                self.writer.add_text(tag="Task List", text_string=f"Task {task_seq}: " + ", ".join(task), global_step=task_seq)


        e_digit = len(str(n_epcoh))
        ee_digit = len(str(early_stop))
        t_digit = len(str(len(task_list)))
        torch.autograd.set_detect_anomaly(True)

        last_best: Union[iCaLRNet, None] = None
        # loaders
        train_loader = data.DataLoader(
            self.train_set, batch_size=bsize, shuffle=True, num_workers=self.num_worker, pin_memory=True
        )
        val_loader = data.DataLoader(
            self.val_set, batch_size=bsize, shuffle=False, num_workers=self.num_worker, pin_memory=True
        )

        # Attention: Pytorch Code Implementation of Algorithm 2: iCaRL INCREMENTALTRAIN
        for task_idx, task in enumerate(task_list):
            x: torch.Tensor
            y: torch.Tensor
            loss: torch.Tensor

            self.logger.info(f"{Fore.BLUE}New Task: {task}")
            
            self.net.set_task(task=task)
            self.train_set.set_task(task=task)

            classification_evaluator = ClassificationEvaluator(num_class=len(task))

            # Attention: Algorithm 3: iCaRL UPDATEREPRESENTATION
            # set q
            if task_idx >= 1:
                with torch.no_grad():
                    for seen_class in self.train_set.examplar_set.examplar_set.keys():
                        x = torch.from_numpy(self.train_set.examplar_set.examplar_set[seen_class]["x"])
                        x = x.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                        # dereference self.net(x)
                        q = self.net(x)[:][:,:len(self.net.seen_classes)-len(task)]
                        self.train_set.examplar_set.update_q(class_name=seen_class, q=q)
            
            # learn new task
            max_top1_acc: float = 0
            early_stop_cnt: float = 0
            for epoch in range(n_epcoh):
                # train
                self.net.train()
                for idx, x, y in train_loader:
                    x = x.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                    y = y.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                    y_pred = self.net(x)
                    loss: Union[torch.Tensor, Tuple[torch.Tensor]] = self.net.loss_func(y_pred=y_pred, y=y)

                    # dont forget
                    self.optim.zero_grad()

                    if isinstance(loss, tuple):
                        loss[0].backward(retain_graph=True)
                        loss[1].backward()
                        if not self.dry_run:
                            self.writer.add_scalar(tag="Loss/CrossEntropy", global_step=loss_step, scalar_value=loss[0].cpu().item())
                            self.writer.add_scalar(tag="Loss/Distillation", global_step=loss_step, scalar_value=loss[1].cpu().item())
                            loss_step += 1
                    else:
                        loss.backward()
                        if not self.dry_run:
                            self.writer.add_scalar(tag="Loss/CrossEntropy", global_step=loss_step, scalar_value=loss.cpu().item())
                            loss_step += 1

                    self.optim.step()
                    # record example in the first epoch
                    if epoch == 0:
                        self.train_set.examplar_set.add_batch(x=x, y=y)
                
                # val
                self.net.eval()
                self.val_set.set_task(task=task)
                classification_evaluator.new_epoch()
                acc_num = 0
                all_num = 0
                with torch.no_grad():
                    for idx, x, y in val_loader:
                        x = x.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                        y = y.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)

                        # Warning 有问题, Examplar set 一开始取得不准
                        temp_examplar_set = net.examplar_set.construct_examplar_set(phi = self.net, temp=True)
                        y_classify = self.net.inference(x=x, temp_examplar_set=temp_examplar_set)

                        # classification evaluates
                        all_num += len(y)
                        acc_num += sum(y[:, 0] == y_classify[:, 0])

                top1_current_acc = acc_num / all_num
                # save checkpoint
                if top1_current_acc > max_top1_acc:
                    max_top1_acc = top1_current_acc
                    early_stop_cnt = 0
                    last_best = copy.deepcopy(self.net.state_dict())
                    self.logger.info(f"{Fore.GREEN}Task: [{task_idx:>{t_digit}}|{task}], Epoch: [{epoch:>{e_digit}}/{n_epcoh}], new top1 Acc: {top1_current_acc:>.5f}")
                    if not self.dry_run:
                        torch.save(self.net, (f:=self.checkpoint_path / f"task_{task_idx}-best.pt"))
                else:
                    early_stop_cnt += 1
                    self.logger.info(f"{Fore.GREEN}Task: [{task_idx:>{t_digit}}|{task}], Epoch: [{epoch:>{e_digit}}/{n_epcoh}], Acc: [{top1_current_acc:>.5f}|{max_top1_acc:>.5f}], Early Stop: [{early_stop_cnt:>{ee_digit}}|{early_stop}]")
                
                # writer log task acc
                if not self.dry_run:
                    self.writer.add_scalars(
                        main_tag=f"Task_{task_idx}-Acc",
                        tag_scalar_dict={
                            "mAcc-top1": top1_current_acc,
                            "mAcc-max-top1": max_top1_acc
                        },
                        global_step=epoch
                    )

                if early_stop_cnt >= early_stop:
                    break

            # load best
            self.net.load_state_dict(last_best)
            self.logger.info(f"{Fore.GREEN}Task: [{task_idx:>{t_digit}}|{task}], end at Epoch: [{epoch:>{e_digit}}/{n_epcoh}], switch to best model, best Acc: [{max_top1_acc:>.5f}]")

            # Attention: construct examplar set, Algorithm 4: iCaRL CONSTRUCTEXEMPLARSET
            self.train_set.examplar_set.construct_examplar_set(phi=self.net)

            # Attention: reduce examplar set, Algorithm 5: iCaRL REDUCEEXEMPLARSET
            self.train_set.examplar_set.reduce_examplar_set()
            

            # test cl performance
            task_acc = []
            for past_task_idx, past_task in enumerate(task_list[:task_idx+1]):
                p_all_num = 0
                p_acc_num = 0
                with torch.no_grad():
                    self.val_set.set_task(task=past_task)
                    for idx, x, y in val_loader:
                        x = x.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                        y = y.to(device=self.available_device, dtype=self.default_dtype, non_blocking=True)
                        y_classify = self.net.inference(x=x)

                        # log
                        p_all_num += len(y)
                        p_acc_num += sum(y[:, 0] == y_classify[:, 0])
                task_acc.append(p_acc_num / p_all_num)
                self.logger.info(f"{Fore.YELLOW}Past Task: [{past_task_idx:>{task_idx}}|{past_task}], Acc: {task_acc[-1]:>.5f}")
            acc = sum(task_acc) / len(task_acc)
            self.logger.info(f"{Fore.MAGENTA}After Task: [{task_idx:>{t_digit}}|{task}], CL_Acc: {acc:>.5f}")

            if not self.dry_run:
                scalar_dict: Dict[str, float] = {
                    f"Per-Task mAcc/Task_{past_task_idx}":past_task_acc for past_task_idx, past_task_acc in enumerate(task_acc)
                }
                self.writer.add_scalars(main_tag=f"CL Evaluation", tag_scalar_dict=scalar_dict, global_step=task_idx)
                self.writer.add_scalar(tag=f"CL Evaluation/Mean Task Acc", scalar_value=acc, global_step=task_idx)

        return self.net



if __name__ == "__main__":
    # Notes: Parameters
    if_shuffle = True
    num_class_per_task = 2
    
    # Bug?: 在训练到后面的任务的时候, cl的平均acc就成0了

    task_list = []
    if if_shuffle:
        cifar100_labels.shuffle()

    for i in range(0, len(cifar100_labels.cifar100_labels), num_class_per_task):
        task_list.append(tuple(cifar100_labels.cifar100_labels[i: i+num_class_per_task]))

    e = ExamplarSet()
    net = iCaLRNet(num_class=100, target_dataset="Cifar100", examplar_set=e)
    net.__fe__ += "32"
    trainer = CLTrainer(network=net, task_list=task_list, dry_run=False, log=True)
    try:
        net = trainer.train(n_epcoh=70, early_stop=5)
    except KeyboardInterrupt:
        del trainer, net

    # debug, record the GPU memory use
    # from pytorch_memlab import LineProfiler
    # trainer = CLTrainer(network=net, dry_run=True)
    # try:
    #     with LineProfiler(trainer.train) as prof:
    #         trainer.train(task_list=task_list, n_epcoh=6)
    # except RuntimeError:
    #     prof.print_stats()



