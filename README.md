<div align="center">
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-culane)](https://paperswithcode.com/sota/lane-detection-on-culane?p=clrnet-cross-layer-refinement-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-llamas)](https://paperswithcode.com/sota/lane-detection-on-llamas?p=clrnet-cross-layer-refinement-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clrnet-cross-layer-refinement-network-for/lane-detection-on-tusimple)](https://paperswithcode.com/sota/lane-detection-on-tusimple?p=clrnet-cross-layer-refinement-network-for)


</div>


<div align="center">

# [RE] CLRNet: Cross Layer Refinement Network for Lane Detection

</div>



This repository is the PyTorch and PyTorch Lightning implementation of the paper ["CLRNet: Cross Layer Refinement Network for Lane Detection"](https://arxiv.org/abs/2203.10350). The paper aims to improve the efficacy of lane detection by utilizing existing techniques as a foundation.<br> Previous lane detection approaches have relied heavily on parameter-based techniques and pre-defined features, such as anchor points and segmentation maps. In contrast, the network introduced in this paper classifies features into two broad categories: high-level and low-level.<br>Using a novel loss function **Line IoU loss** improvements in the performance of the network is achieved.

## Model Architecture
![Arch](.github/arch.png)
- CLRNet exploits more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. 
- CLRNet achieves SOTA result on CULane, Tusimple, and LLAMAS datasets.

## Installation

### Prerequisites
- Python >= 3.8 (tested with Python 3.8)
- PyTorch >= 1.6 (tested with Pytorch 1.6)
- WandB >= 0.13.9 (tested with WandB 0.13.9)
- CUDA (tested with cuda11.4)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/KaushalJadhav/clrnet
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3.8 -y
conda activate clrnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://www.kaggle.com/datasets/manideep1108/culane). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```


#### Tusimple
Download [Tusimple](https://www.kaggle.com/datasets/manideep1108/tusimple). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

#### LLAMAS
Dowload [LLAMAS](https://unsupervised-llamas.com/llamas/). Then extract them to `$LLAMASROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $LLAMASROOT data/llamas
```

Unzip both files (`color_images.zip` and `labels.zip`) into the same directory (e.g., `data/llamas/`), which will be the dataset's root. For LLAMAS, you should have structure like this:
```
$LLAMASROOT/color_images/train # data folders
$LLAMASROOT/color_images/test # data folders
$LLAMASROOT/color_images/valid # data folders
$LLAMASROOT/labels/train # labels folders
$LLAMASROOT/labels/valid # labels folders
```


## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, for DLA34 backbone and CULane dataset run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --gpus 0
```

### Validation
For validtion, run
```Shell
python main.py [configs/path_to_your_config] --[validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, for DLA34 backbone and CULane dataset run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

### Testing
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, for DLA34 backbone and CULane dataset run
```Shell
python main.py configs/clrnet/clr_dla34_culane.py --test --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

## Pre-trained Models

- Pretrained models: [Link](https://drive.google.com/drive/folders/1Zoi2FeB-FdgAXr5owqYeETvVh0YCvKPm?usp=sharing)

## Results

### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 |
| :---  |  :---:   |   :---:    | :---:|
| [ResNet-18][assets]     | 55.23  |  79.58   | 62.21 |
| [ResNet-34][assets]     | 55.14  |  79.73   | 62.11 |
| [ResNet-101][assets]     | 55.55| 80.13   | 62.96 |
| [DLA-34][assets]     | 55.64|  80.47   | 62.78 |



### TuSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][assets]     |    97.89    |   96.84  |    2.28  |  1.92      | 
| [ResNet-34][assets]       |   97.82              |    96.87          |   2.27          |    2.08      | 
| [ResNet-101][assets]      |   97.62|   96.83  |   2.37   |  2.38  |


## Bibtex

- CLRNet: Cross Layer Refinement Network for Lane Detection
    ```
    @InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
    }
    ```


## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->
