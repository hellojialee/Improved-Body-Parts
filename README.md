# SimplePose

Code and pre-trained models for our paper.
The old repo which can be referred is [**Here**](https://github.com/jialee93/Multi-Person-Pose-using-Body-Parts).

## Introduction

A bottom-up approach for the problem of multi-person pose estimation.
The  source code will be improved as soon as the paper is accepted and more instructions will be added.

### Contents

1. Training 
2. Evaluation 
3. Demo

### Task Lists
- [ ] Rewrite and speed up the code of keypoint assignment in C++  with multiple processes.


## Project Features
- Implement the models using Pytorch in auto mixed-precision.
- Supprot training on multiple GPUs (over 90% GPU usage rate on each GPU card).
- Fast data preparing and augmentation during training.

## Prepare

1. Install packages:

   Python=3.6, Pytorch>1.0, Nvidia Apex and other packages needed.

2. Download the COCO dataset.

3. Download the pre-trained models.

   According to AAAI Blind Review Instructions:

   > Submissions should not contain pointers to supplemental material on the web.

   Download Link: we will complete this after AAAI review period.

4. Change the paths in the code according to your environment.

## Run a Demo

`python demo_image.py`

## Evaluation Steps

The corresponding code is in pure python without multiprocess for now.

`python evaluate.py` 

## Training Steps

Before training, prepare the training data using ''SimplePose/data/coco_masks_hdf5.py''.

Multiple GUPs are recommended to use to speed up the training process, but we support different training options. 

- [x] Most code has been provided already, you can train the model with.

  1.  'train.py': single training process on one GPU only.
  2.  'train_parallel.py': signle training process on multiple GPUs using Dataparallel.
  3.  'train_distributed.py' (**recommended**): multiple training processes on multiple GPUs using Distributed Training:

```shell
python -m torch.distributed.launch --nproc_per_node=4 train_distributed.py
```



## Referred Repositories (mainly)

-  [Realtime Multi-Person Pose Estimation verson 1](we will complete this link after AAAI review period)
-  [Realtime Multi-Person Pose Estimation verson 2](we will complete this link after AAAI review period)
-  [Realtime Multi-Person Pose Estimation version 3](we will complete this link after AAAI review period)
-  [Associative Embedding](we will complete this link after AAAI review period)
-  [NVIDIA/apex](we will complete this link after AAAI review period)
