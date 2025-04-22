# TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals, ICRA 2025

[//]: # (<img width="806" alt="image" src="./assets/pixelnav.png">)

This is the official implementation of the paper. Please refer to the [paper](...) and 
[website](https://podgorki.github.io/TANGO/) for more technique details.

This repository contains the TANGO controller and a minimal setup required to show the demo on habitat-sim. 

This work is based off a larger project beginning with RoboHop and the full evaluation for this code and Robohop can be 
found at [website](...) 

## Installation 
### Create new environment

```commandline
python3.10 -m venv .venv --prompt tango
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Install controller and sim (required for demo)

### Pre-install habitat-sim

#### Dependencies
```commandline
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
pip install cmake==3.14.4
pip install "numpy>=1.25,<2" --upgrade  # required before building habitat-sim
```

#### Build the Sim (takes a bit)
```commandline
cd third-party/
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim/
git checkout v0.2.4
python setup.py install --cmake
cd ../..
```

### Install TANGO

```commandLine
pip install -e ".[habitat-lab]" --extra-index-url https://download.pytorch.org/whl/cu128 --prefer-binary
``` 

### Depth anything
Depth anything is installed by submoduling.

Add a pth so you can resolve zoedepth
```commandline
echo "$PWD/third_party/depth_anything/metric_depth" > \
     $(python -c "import site, sys; print(site.getsitepackages()[0])")/zoedepth_local.pth
```

The depth anything model weights are located at: https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth
And also grab the vit from here https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints
place them in third_party/models/

## TANGO Demo
Update the config yaml with the dataset locations
```commandline
python -m scripts.run_goal_control_demo
```

## BibTex
Please cite our paper if you find it helpful :)
```

[//]: # (@inproceedings{cai2024bridging,)

[//]: # (  title={Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill},)

[//]: # (  author={Cai, Wenzhe and Huang, Siyuan and Cheng, Guangran and Long, Yuxing and Gao, Peng and Sun, Changyin and Dong, Hao},)

[//]: # (  booktitle={2024 IEEE International Conference on Robotics and Automation &#40;ICRA&#41;},)

[//]: # (  pages={5228--5234},)

[//]: # (  year={2024},)

[//]: # (  organization={IEEE})

[//]: # (})
