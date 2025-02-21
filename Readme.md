## Setup
```
conda create -n tango-habitat python=3.9 cmake=3.14.0
conda activate tango-habitat

# install habitat-lab
cd libs/
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout v0.2.4
pip install -e habitat-lab

# Download SegmentMap repository from https://universityofadelaide.box.com/s/rbge9q7mv0alucvqjelx8odisw59mehm and place it in .`/auto_agents/`

# install RoboHop dependencies (assumed SegmentMap folder is already copied over or downloaded from github)
cd auto_agents/SegmentMap/sam
pip install -e . 

# Download SAM (ViT-H) checkpoint from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints and place it in `./models/segment-anything/` (or use symlinks)

# clone AnyLoc for DINOv2 features
cd auto_agents/SegmentMap
git clone https://github.com/AnyLoc/AnyLoc.git
```
### Examples
#### Downloads
Please first download image trajectory data from https://universityofadelaide.box.com/s/ahjnfds5xtjn2xh1f3f9rei8d8to418e

You can also download the precomputed data (~2.4 GB, DINOv2 features and SAM masks for the above trajectory map) from https://universityofadelaide.box.com/s/zzj02dsz6qhaig4v8zd8ow0kqjxc26y6 and place contents of this folder directly in `./out/RoboHop/` (this path is set through `h5FullPath` argument when instantiating RoboHop)

#### Example Navigation
The following will first create a map and a graph (if precomputed data is not being used), and then run a navigation episode (output logs are stored in `./out/runs/dump/.../controlLogs/matchings/`)
```
# seed reproduces specific nav episodes
python teleop.py --seed 6

# for using weights when computing path lengths (The specified string must be an edge attribute in the topological graph, e.g., see https://github.com/oravus/sg_habitat/blob/10e1a373e1f930184da9ffb167e3df8894c20bcc/auto_agent.py#L295)
python teleop.py --seed 6 --weight_string "margin"
```

#### Example Notebook
See [tests/test_RH_skokloster.ipynb](./tests/test_RH_skokloster.ipynb)


#### Example Habitat Sim Teleop w Semantic 
```
cd <repo_dir>

# Download sample HM3D Data (84 MB)
wget -O hm3d_00853-5cdEh9F2hJL.zip "https://universityofadelaide.app.box.com/index.php?rm=box_download_shared_file&shared_name=6w9kxhybckhh3883i9ybtc09ulmk9nuc&file_id=f_1562632791659"
unzip hm3d_00853-5cdEh9F2hJL.zip

# teleop using wad or arrows from terminal & display will update
python scripts/hab_semantic_teleop.py

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
