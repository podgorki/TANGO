## Setup | Conda/Mamba | NO SAM2.1

```
conda create -n nav
conda activate nav

conda install python=3.9 mamba -c conda-forge
mamba install numpy matplotlib pip pytorch torchvision pytorch-cuda=11.8 opencv=4.6 cmake=3.14.0 habitat-sim withbullet numba=0.57 pyyaml ipykernel networkx h5py natsort open-clip-torch transformers einops scikit-learn kornia pgmpy python-igraph pyvis -c pytorch -c nvidia -c aihabitat -c conda-forge

[optional] mamba install -c conda-forge tyro faiss-gpu scikit-image ipykernel spatialmath-python gdown utm seaborn wandb kaggle yacs
[optional] mamba install -c conda-forge ultralytics

# install habitat-lab
cd libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab


```

## Setup | Pip | WITH SAM2.1

```
conda create -p ~/envs/habsam2
conda activate ~/envs/habsam2
conda install -c conda-forge python=3.10 # Need >=3.10 (sam2.1), <=3.10 (habitat_sim) 

# if nvidia drivers version < 525.60.13, then install torch first before installing SAM2.1
pip install torch torchvision --force-reinstall --pre --extra-index-url https://download.pytorch.org/whl/cu118

git clone git@github.com:oravus/sg_habitat.git
cd sg_habitat/

# clone sam2 into libs (remove any prior sam2 installation) 
rm -r libs/segmentor/sam2
git rm --cached libs/segmentor/sam2
git submodule add https://github.com/facebookresearch/sam2.git libs/segmentor/sam2

# install sam2 and its dependencies
cd libs/segmentor/sam2/
pip install -e .

# install other dependencies
pip install opencv-python networkx h5py natsort einops scikit-learn kornia pgmpy python-igraph pyvis tyro scikit-image ipykernel spatialmath-python gdown utm seaborn wandb kaggle yacs cmake==3.14.3

# install prerequisites for habitat-sim
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev

# download (anywhere) and compile habitat-sim
cd sg_habitat/libs/
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim/
git checkout v0.2.4
python setup.py install --cmake

# download and install habitat-lab
cd sg_habitat/libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab

# models factories
pip install huggingface_hub==0.25.0 ultralytics

# dependencies for visualnav-transformer
pip install warmup-scheduler diffusers==0.11.1 efficientnet_pytorch vit_pytorch lmdb prettytable

```

## Data

- Download official [hm3d v0.2](https://github.com/matterport/habitat-matterport-3dresearch) following their
  instructions.
- Download official `InstanceImageNav` challenge dataset
  from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip) (
  Direct Link | ~512 mb)

Using the above, we generate train and val episodes (i.e., images along trajectories), which can be downloaded as below
for testing and training.

### Test Navigation

- Download `hm3d_iin_val` trajectory data
  from [here](https://universityofadelaide.box.com/s/j4chd1uux1omyiscp544b26z5wjlt95d).
- Download `robohop.yaml` and `gnm_test.yaml` (if testing learnt controller)
  from [config files dir](https://universityofadelaide.box.com/s/hj5bmb81v2h1zpllaw2ib2lk3t8g6bfa) to
  `sg_habitat/configs/`

Run
`python goal_control.py --path_dataset <path to parent dir of all downloaded/symlinked dirs> --split val --method robohop`

### Train Controller

- Download `hm3d_iin_train` trajectory data
  from [here](https://universityofadelaide.box.com/s/cch6r0ue7z377q79g2j4vvemnq3ov8rg).
- Download `gnm_sg.yaml`
  from [the same config files dir](https://universityofadelaide.box.com/s/hj5bmb81v2h1zpllaw2ib2lk3t8g6bfa) to
  `visualnav-transformer/train/config/gnm_sg.yaml`
- Download `data_config.yaml`
  from [the same config files dir](https://universityofadelaide.box.com/s/hj5bmb81v2h1zpllaw2ib2lk3t8g6bfa) to
  `visualnav-transformer/train/vint_train/data/data_config.yaml`
- Download precomputed h5 file
  for [gt_topometric masks](https://universityofadelaide.box.com/s/oy9z372e8i4a2rsrn78agpaploljpas4)
  or [sam21 masks](https://universityofadelaide.box.com/s/5z2sr6lvkax2l160vp8rz5iuqx8x7ogz)
- ~~Download data splits file from [here]()~~

## IGNORE Old Instructions below

```
# Download SegmentMap repository from https://universityofadelaide.box.com/s/rbge9q7mv0alucvqjelx8odisw59mehm and place it in .`/auto_agents/`

# install RoboHop dependencies (assumed SegmentMap folder is already copied over or downloaded from github)
cd auto_agents/SegmentMap/sam
pip install -e . 

# Download SAM (ViT-H) checkpoint from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints and place it in `./models/segment-anything/` (or use symlinks)

# install SAM2 (from the local lib directory)
cd libs/segmentor/segment_anything_2
pip install -v -e ".[demo]"
# Note: to force building the CUDA extension, set variable `SAM2_BUILD_ALLOW_ERRORS=0`.
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[demo]"
# then to download the checkpoints ...
cd checkpoints
./download_ckpts.sh 


# clone AnyLoc for DINOv2 features
cd auto_agents/SegmentMap
git clone https://github.com/AnyLoc/AnyLoc.git
```

### Examples

#### Downloads

Please first download image trajectory data from https://universityofadelaide.box.com/s/ahjnfds5xtjn2xh1f3f9rei8d8to418e

You can also download the precomputed data (~2.4 GB, DINOv2 features and SAM masks for the above trajectory map)
from https://universityofadelaide.box.com/s/zzj02dsz6qhaig4v8zd8ow0kqjxc26y6 and place contents of this folder directly
in `./out/RoboHop/` (this path is set through `h5FullPath` argument when instantiating RoboHop)

#### Example Navigation

The following will first create a map and a graph (if precomputed data is not being used), and then run a navigation
episode (output logs are stored in `./out/runs/dump/.../controlLogs/matchings/`)

```
# seed reproduces specific nav episodes
python teleop.py --seed 6

# for using weights when computing path lengths (The specified string must be an edge attribute in the topological graph, e.g., see https://github.com/oravus/sg_habitat/blob/10e1a373e1f930184da9ffb167e3df8894c20bcc/auto_agent.py#L295)
python teleop.py --seed 6 --weight_string "margin"
```

# new setup

## create new env

```commandline
python3.10 -m venv .venv --prompt tango
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## install only the controller with

```commandLine
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118 --prefer-binary
  ```

## install controller and sim (required for demo)

### pre install habitat-sim

```commandline
cd third-party/
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim/
git checkout v0.2.4
python setup.py install --cmake
cd ../..
```

```commandLine
pip install -e ".[habitat-lab]" --extra-index-url https://download.pytorch.org/whl/cu128 --prefer-binary
``` 

#### TANGO Demo

```
cd <repo_dir>

# Download sample HM3D Data (84 MB)
wget -O hm3d_00853-5cdEh9F2hJL.zip "https://universityofadelaide.app.box.com/index.php?rm=box_download_shared_file&shared_name=6w9kxhybckhh3883i9ybtc09ulmk9nuc&file_id=f_1562632791659"
unzip hm3d_00853-5cdEh9F2hJL.zip


```
