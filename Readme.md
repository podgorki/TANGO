# Setup

```
conda create -n tango-habitat python=3.9 cmake=3.14.0
conda activate tango-habitat

# install habitat-lab
cd libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab

# (# Download SegmentMap repository from https://universityofadelaide.box.com/s/rbge9q7mv0alucvqjelx8odisw59mehm and place it in .`/auto_agents/`)

# install RoboHop dependencies (assumed SegmentMap folder is already copied over or downloaded from github)
cd auto_agents/SegmentMap/sam
pip install -e . 

# Download SAM (ViT-H) checkpoint from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints and place it in `./models/segment-anything/` (or use symlinks)

# clone AnyLoc for DINOv2 features
cd auto_agents/SegmentMap
git clone https://github.com/AnyLoc/AnyLoc.git
```

# Pixnav

submodule pixnav with

** git submodule add https://github.com/wzcai99/Pixel-Navigator.git libs/pixnav **

# depth anything

submodule depth anything with

** git submodule add https://github.com/LiheYoung/Depth-Anything.git libs/depth/depth_anything **

add libs.depth.depth_anything.metric_depth. to all zoedepth imports

# lightglue

submodule light glue with

** git submodule add https://github.com/cvg/LightGlue.git libs/matcher/LightGlue **

# segment anything

submodule SAM with

** git submodule add https://github.com/facebookresearch/segment-anything.git libs/segmentor/segment_anything **

then install with

cd libs/segmentor/segment_anything; pip install -e .
