# iCaRL: Incremental Classifier and Representation Learning Pytorch Implementation

This repository is `iCaRL: Incremental Classifier and Representation Learning` Pytorch third-party implementation.

PDF: https://arxiv.org/pdf/1611.07725.pdf

Arxiv: https://arxiv.org/abs/1611.07725

The original repository: https://github.com/srebuffi/iCaRL/tree/7a3d254da4b3f0f19f67b8c25ba72374674bafaf/iCaRL-TheanoLasagne

Notes:

1. The original paper used decrease learning rate and other tricks, like center crop, five crop and etc.. But, for the simplicity, I **didn't** do these tricks.
2. Only `Cifar100` is tested, for I have no access to Stronger GPUs, so it's not realistic for me to test on `ImageNet`.
3. Some modifications are made in order to speed up the training process.

## Requirements

- `Windows11` / `Ubuntu20.04`
- `Python 3.8/3.9/3.10` (Since I love to use `:=`, the python version must be higher than 3.8, but 3.9 is recommended)
- `Pytorch 1.11.0` (But should work properly with previous version)
- `CUDA 11.3`
- `Numpy`, `colorama`, `PIL`, `Pandas`, `matplotlib`, `terminaltables` 

## Run

The **first** thing you need to do is make a soft link to `Cifar100` datasets under the `dataset ` folder. Or, you can modify the `DatasetPath` in the `helper.py` to point to the right folder.

For me, it is

- Windows (Command Prompt)

```powershell
mkdir datasets
cd datasets
mklink /D cifar100-windows <path-to-cifar100>
```

- Ubuntu

```shell
mkdir datasets
cd datasets
ln -s cifar100-linux <path-to-cifar100>
```

Then, the **second** steps is run the command below,

```shell
python3 main.py -h
```

and you will see help information.

```shell
jack@jack:~\projects\iCaLR$ python .\main.py -h

usage: main.py [-h] [-v] [-s] [-d] [-l] [-n NUM_CLASS] [-ne N_EPOCH] [-es EARLY_STOP] [-tm TEST_METHOD]
               [-nt NUM_TASK] [-lf LOSS_FUNC] [-m MESSAGE]

iCaRL Pytorch Implementation training util by Shihong Wang (Jack3Shihong@gmail.com)

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -s, --shuffle         If shuffle the training classes
  -d, --dry_run         If run without saving tensorboard amd network params to runs and checkpoints
  -l, --log             If save terminal output to log
  -n NUM_CLASS, --num_class NUM_CLASS
                        Set class number of each task
  -ne N_EPOCH, --n_epoch N_EPOCH
                        Set maximum training epoch of each task
  -es EARLY_STOP, --early_stop EARLY_STOP
                        Set maximum early stop epoch counts
  -tm TEST_METHOD, --test_method TEST_METHOD
                        Set test method, 0 for classify, 1 for argmax
  -nt NUM_TASK, --num_task NUM_TASK
                        All predict class num, if set to None, will add gradually
  -lf LOSS_FUNC, --loss_func LOSS_FUNC
                        Select loss function type, 1 for other implementation, 0 for my
                        implementation
  -m MESSAGE, --message MESSAGE
                        Training digest
```



Or, you can modify the `experiments.py` to automatically run the main experiment (num_task = 2, 5, 10, 20, 50) in the paper.



## Results

New: `BUG`s find! The reason of why the result is not goold is simply because I didn't train the network well. More expriment are coming soon.

Notes: I tried my best to re-implement all the TRAINING details in the paper and I believe my operation are correct. However, due to the difference in evaluation method, there is still performance gap between my result and original result.

Moreover, I find a [repository](https://github.com/DRSAD/iCaRL/tree/9768d45a86d7b43acfcf539ad84ae6d88f47d9e7 "repository") which totally re-implements all operations including the evaluation method. But after reviewing the codes, I find some KEY STEPS in the evaluation that I do not think is objective. So, I decide not to re-implement the biased performance code.

Since I have invest two weeks in this project and find the biased evaluation, I need to go on and move to next algorithm to implement . I hope someone could find the bugs (if there is in my code) and fix the gap between my code. Feel free to contact me via Jack3Shihong@gmail.com if you find and bugs or have any problem. I'm not sure I can fix your problem, bug I will try my best

The biased performance evaluation code and the reason why I think they are biased in the listed repository are listed below.



You can find all the training results in `./runs` and `./log`.

Run

```shell
tensorboard --logdir ./runs
```

to check all the results. More detailed training results and log can be found in `./log`.



Since all of the parameters in `checkpoints` is so large (~50G), I'm not going to upload the parameters. You can get the parameter by runing

```python
python main.py -l -n NUM_TASK
```

The log, parameter and tensorboard results will be saved in `./checkpoints`, `./log` and `./runs`



### Bias Evaluation

1. I think one should evaluate the performance on all past classes in all past tasks. However, the code in DRSAD's repository only test on new task after learning a new task.
2. The testing code used multiple data augmentation methods, but I don't think it should be done to get a really fancy result since other method may not do this.
3. The training codes, the loss function in the repository mismatch the loss function in the paper. In detail, the distillation loss should only be applied on old classes, instead of all class in DRSAD's repository.




## Illustrations

Since the codes use some tricks to realize the algorithms, some key points in the code will be illustrated in both Chinese and English Version.

Some major points:

- Distillation loss: BCELoss (Binary Cross Entropy Loss)
- Weight Vector: Linear Layer

More detailed explanation is coming...
