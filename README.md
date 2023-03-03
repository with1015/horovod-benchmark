# PyTorch Benchmarks

## Setup
### Software packages
    - CUDA 10.1
    - Nvidia driver 418.56
    - Ubuntu 16.04
    - PyTorch >= 1.1

### Build Conda Environment
```
$ conda create -n pt_1.8 python=3.6
$ conda activate pt_1.8
$ conda install pytorch==1.8.1 torchvision cudatoolkit=10.1 -c pytorch
$ pip install -r requirements.txt
```

#### Thanks to
All code is modified from https://github.com/NVIDIA/DeepLearningExamples/tree/master
