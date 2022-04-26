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

Only thing you need to do is make a soft link to `Cifar100` datasets under the `dataset ` folder.

## Results

Coming Soon...

## Illustrations

Since the codes use some tricks to realize the algorithms, some key points in the code will be illustrated in both Chinese and English Version.

Coming Soon...
