# SimplePose

Code and pre-trained models for our paper, [“Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation”](https://arxiv.org/abs/1911.10529), accepted by AAAI-2020. 

Also this repo serves as the **Part B** of our paper "Multi-Person Pose Estimation using Body Parts" (under review). The **Part A** is available at [this link](https://github.com/jialee93/Multi-Person-Pose-using-Body-Parts).

## Introduction

A bottom-up approach for the problem of multi-person pose estimation.

![heatmap](visulizatoin/2987.Figure2.png)

![network](visulizatoin/2987.Figure3.png)

### Contents

1. Training  
2. Evaluation 
3. Demo

## Project Features

- Implement the models using Pytorch in auto mixed-precision.
- Supprot training on multiple GPUs (over 90% GPU usage rate on each GPU card).
- Fast data preparing and augmentation during training.
- **This project can also serve as a detailed practice to the green hand in Pytorch.**

## Prepare

1. Install packages:

   Python=3.6, Pytorch>1.0, Nvidia Apex and other packages needed.

2. Download the COCO dataset.

3. Download the pre-trained models (default configuration: download the pretrained model snapshotted at epoch 52 provided as follow).

   Download Link: [BaiduCloud](https://pan.baidu.com/s/1X7nGC-7CliP1iKgIfsBMUg)

4. Change the paths in the code according to your environment.

## Run a Demo

`python demo_image.py`

## Inference Speed

The speed of our system is tested on the MS-COCO test-dev dataset. 

- Inference speed of our 4-stage IMHN with 512 × 512 input on one 2080TI GPU: 38.5 FPS (100% GPU-Util). 
- Processing speed of the keypoint assignment algorithm part that is implemented in pure Python and a single process on Intel Xeon E5-2620 CPU: 5.2 FPS (has not been well accelerated). 

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

- [Realtime Multi-Person Pose Estimation verson 1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation verson 2](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation version 3](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
- [Associative Embedding](https://github.com/princeton-vl/pose-ae-train)
- [NVIDIA/apex](https://github.com/NVIDIA/apex)

## Citation

Please kindly cite this paper in your publications if it helps your research.

```
@misc{li2019simple,
    title={Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation},
    author={Jia Li and Wen Su and Zengfu Wang},
    year={2019},
    eprint={1911.10529},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

