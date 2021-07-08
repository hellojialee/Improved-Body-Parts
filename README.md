# SimplePose

Code and pre-trained models for our paper, [“Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation”](https://arxiv.org/abs/1911.10529), accepted by AAAI-2020. 

Also this repo serves as the **Part B** of our paper "Multi-Person Pose Estimation Based on Gaussian Response Heatmaps" (under review). The **Part A** is available at [this link](https://github.com/jialee93/Multi-Person-Pose-using-Body-Parts).

- [x] Update

  A faster project is released at [OffsetGuided](https://github.com/hellojialee/OffsetGuided)
  
  A Chinese note at [Zhihu](https://zhuanlan.zhihu.com/p/109118177)



## Introduction

A bottom-up approach for the problem of multi-person pose estimation.

![heatmap](visulizatoin/2987.Figure2.png)

![network](visulizatoin/2987.Figure3.png)

### Contents

1. Training  
2. Evaluation 
3. Demo

## Project Features

- Implement the models using Pytorch in auto mixed-precision (using Nvidia Apex).
- Support training on multiple GPUs (over 90% GPU usage rate on each GPU card).
- Fast data preparing and augmentation during training (generating about **40 samples per second** on signle CPU process and much more if wrapped by DataLoader Class).
- Focal L2 loss.
![FL2](visulizatoin/fl2.png)
- Multi-scale supervision.
- **This project can also serve as a detailed practice to the green hand in Pytorch.**

## Prepare

1. Install packages:

   Python=3.6, Pytorch>1.0, Nvidia Apex and other packages needed.

2. Download the COCO dataset.

3. Download the pre-trained models (default configuration: download the pretrained model snapshotted at epoch 52 provided as follow).

   Download Link: [BaiduCloud](https://pan.baidu.com/s/1X7nGC-7CliP1iKgIfsBMUg)

   Alternatively, download the pre-trained model without optimizer checkpoint only for the default configuration via [GoogleDrive](https://drive.google.com/open?id=1gLa2oNxnbFPo0BjnpPaiAmWJwyND8wkA)

4. Change the paths in the code according to your environment.

## Run a Demo

`python demo_image.py`

![examples](visulizatoin/examples.png)

## Inference Speed

The speed of our system is tested on the MS-COCO test-dev dataset. 

- Inference speed of our 4-stage IMHN with 512 × 512 input on one 2080TI GPU: 38.5 FPS (100% GPU-Util). 
- Processing speed of the keypoint assignment algorithm part that is implemented in pure Python and a single process on Intel Xeon E5-2620 CPU: 5.2 FPS (has not been well accelerated). 

## Evaluation Steps

The corresponding code is in pure python without multiprocess for now.

`python evaluate.py` 

Results on MSCOCO 2017 test-dev subset (focal L2 loss with gamma=2):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.685
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.867
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.749
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.664
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.892
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.688
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.784
```

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

> Note:  The *loss_model_parrel.py* is for *train.py* and *train_parallel.py*, while the *loss_model.py* is for *train_distributed.py* and *train_distributed_SWA.py.* They are different in dividing the batch size. Please refer to the code about the different choices. 
>
> For distributed training, the real batch_size = batch_size_in_config* × GPU_Num (world_size actually). For others, the real batch_size = batch_size_in_config*. The differences come from the different mechanisms of data parallel training and distributed training. 

## Referred Repositories (mainly)

- [Realtime Multi-Person Pose Estimation version 1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation version 2](https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation)
- [Realtime Multi-Person Pose Estimation version 3](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
- [Associative Embedding](https://github.com/princeton-vl/pose-ae-train)
- [NVIDIA/apex](https://github.com/NVIDIA/apex)

## Recommend Repositories

[Faster Version](https://github.com/sokunmin/Improved-Body-Parts): Chun-Ming Su has rebuilt and improved the post-processing speed of this repo using C++, and the improved system can run up to 7~8 FPS using a single scale with flipping on a 2080 TI GPU. Many thanks to Chun-Ming Su.

## Citation

Please kindly cite this paper in your publications if it helps your research.

```
@inproceedings{li2020simple,
  title={Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation.},
  author={Li, Jia and Su, Wen and Wang, Zengfu},
  booktitle={AAAI},
  pages={11354--11361},
  year={2020}
}
```
