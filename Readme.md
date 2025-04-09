# Setup

```
conda create -n tango-habitat python=3.9 cmake=3.14.0
conda activate tango-habitat
conda install open-clip-torch
```

# Habitat-lab

submodule habitat-lab

```
git submodule add https://github.com/facebookresearch/habitat-lab.git libs/habitat-lab
cd libs/habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab
```


# Pixnav

submodule pixnav with

```
git submodule add https://github.com/wzcai99/Pixel-Navigator.git libs/pixnav
```

# depth anything

submodule depth anything with

```
git submodule add https://github.com/LiheYoung/Depth-Anything.git libs/depth/depth_anything
```

add libs.depth.depth_anything.metric_depth. to all zoedepth imports

# lightglue

submodule light glue with

```
git submodule add https://github.com/cvg/LightGlue.git libs/matcher/LightGlue
```

# segment anything

submodule SAM with

```
git submodule add https://github.com/facebookresearch/segment-anything.git libs/segmentor/segment_anything
```

then install with

```
cd libs/segmentor/segment_anything; pip install -e .
```
