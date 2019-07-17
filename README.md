---
title: 'Project documentation template'
disqus: hackmd
---

2D Bounding Box and Segmentation for Kitti
===
Code based on MMDetection v0.6.0 (14/04/2019): https://github.com/open-mmlab/mmdetection

Model based on mmdet/models/detectors/cascade_rcnn.py

Data from: http://www.cvlibs.net/datasets/kitti/eval_3dobject.php


## Table of Contents

[TOC]

## What is this for?

This is the first part of the proposed framework for 3D object detection. The Cascade R-CNN model in this part of the framework provides model training and testing of 2D bounding box detection on the Kitti dataset. It also provides testing of instance segmentation on the Kitti images using the pretrained model from ([model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth))


## Prerequisites
#### System Requirements
---
1. Linux
2. Python 3.7
4. CUDA 10.0
5. NCCL 2+
6. GCC 4.9+


#### Start the environment
---
Start a conda environment in python 3.7
Activate it
Install Cython

```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install cython
```

Install mmcv
Install pytorch + torchvision and its dependencies

```bash
pip install mmcv
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Run the following commands in mmdetection to finish installation
```bash
./compile.sh
python setup.py develop
# 'pip install -e' works too apparently
```

#### Convert Data from Kitti label format to COCO JSON format
To convert from KITTI training dataset format and labels to a COCO JSON call cocogenesis.py 

```bash
python cocogenesis.py --imagedir=${IMAGE_DIR} --annopath=${ANNO_DIR}
```

## Expected Directory Structure
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── kitti
│   │   ├── annotations
│   │   ├── $path to images$


## Config file overview

Before training or testing, the config file must to defined in configs/ for each specific task (2D bounding box detector and/or segmentation), and for each dataset. The path to the config file will be used to run the model for training or testing.

### Training (Kitti 2D Bounding Box)

The config file is located at configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti.py

The config file was modified to provide the following information:
* bbox_head/num_classes=10, # kitti has 9 classes, so 9 + 1
* img_norm_cfg=... # provide the mean and std of each channel in BGR format from the kitti data for normalization (use the script at ??? to get the mean and std)
* imgs_per_gpu=4     # batch size for training 
* workers_per_gpu=4  # match batch size
* dataset_type = 'KittiDataset' # this refers to datasets/kitti.py
* data_root = 'data/kitti/' # data directory, everything is relative to the mmdetection
* ann_file=data_root + 'annotations/instances_runtrain.json', # path to the train labels in coco format
* img_prefix= data_root + 'training/image_2', # path to the image directory

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001) #manually adjust to lower learning rate as it begins training begins to stagnate 

Command line to start training:
``` bash
./tools/dist_train.sh configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti.py ${GPU_NUM} --validate 
```
where ${GPU_NUM} is the number of gpus to train on.

## Testing (2D Bounding Box)

Use the same config file as for training
Command line to start testing/inferencing
```bash
./tools/dist_test.sh configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti.py ${WEIGHTS_PATH} ${GPU_NUM} --out ${OUTPUT_PATH} --eval bbox 
```
${WEIGHTS_PATH} is the path to the weights that are trained. If the model was trained, the path should be in `work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_kitti/latest.pth`

${OUTPUT_PATH} should have a file extension that would represent a pickle file

## Testing (Instance Segmentation)

The config file is located at configs/htc_mask.py
The config was modified for use to generate segmentation and additional bounding boxes for our dataset
Command Line to start inference
```bash
./tools/dist_test/sh configs/htc_mask.py ${WEIGHTS_PATH} ${GPU_NUM} -- ${OUTPUT_PATH} --eval bbox seg
```
where weights path should be the path to the instance segmentation weights, it is in (what is this for?)

## Using another dataset
in mmdet/datasets/, you will need to create a custom dataset, based on the format you defined it on. 
additionally, in the config file, you need to set and change the data dictionary based on the data it works on

## Post Processing the Testing Output
For the bounding boxes, the main output will be in the format ${OUTPUT_PATH}.output. Use this pickle file to pass to the next step.

The pickle file is a list , length of all predicted bounding boxes of the dataset (there is an average 10 or so boxes per image, with a maximum of 100 boxes), of dictionaries.
Each dictionary has the keys: 'image_id','bbox','score','category_id','conv'
Pass this to pickleslicer to break it up into a format that fusion prefers

How to read the pickle file?
```python
import pickle as pkl
file=pkl.load(open("${PICKLEFILE}",'rb'))
```


What?
Yeah I dont know why we preprocessed for a preprocessor for a preprocessor.
WHy?
We didn't agree on a format that was flexible ok? Stop asking questions. 
