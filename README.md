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
- `Numpy`, `colorama`, `PIL`, `Pandas`, `matplotlib`, `terminaltables` (Unused libraries will be removed later)

## Run

The **first** thing you need to do is make a soft link to `Cifar100` datasets under the `dataset ` folder.

For me, it is

- Windows

```shell
C:\Users\JackWang\Projects\iCaRL> mkdir datasets
C:\Users\JackWang\Projects\iCaRL> cd datasets
C:\Users\JackWang\Projects\iCaRL> mklink /D cifar100-windows <path-to-cifar100>
```

- Ubuntu

```shell
jack@Jack:~\Projects\iCaRL$ mkdir datasets
jack@Jack:~\Projects\iCaRL$ cd datasets
jack@Jack:~\Projects\iCaRL$ ln -s cifar100-linux <path-to-cifar100>
```

Then, the **second** steps is run the command below, and you will see the information.

```shell
python3 main.py -h
```

## Results

Notes: I tried my best to re-implement all the details in the paper and I believe my operation are correct. However, due to the difference in evaluation method, there is still performance gap between my result and original result.

Moreover, I find a [repository](https://github.com/DRSAD/iCaRL/tree/9768d45a86d7b43acfcf539ad84ae6d88f47d9e7 "repository") which totally re-implements all operations including the evaluation method. But after reviewing the codes, I find some KEY STEPS in the evaluation that I do not think is objective. So, I decide not to re-implement the biased performance code.


Since I have invest two weeks in this project and find the biased evaluation, I need to go on and move to next algorithm to implement . I hope someone could find the bugs (if there is in my code) and fix the gap between my code.


The biased performace evaluation code and the reseaon why I think they are biased in the listed repository are listed below.


## Illustrations

Since the codes use some tricks to realize the algorithms, some key points in the code will be illustrated in both Chinese and English Version.

Coming Soon...
