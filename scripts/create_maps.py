 
# %%
# USAGE:
# bash: for i in {0..141}; do python scripts/create_maps.py -c '' -d -i $i; done
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import curses
from importlib import reload
import os
import sys
from natsort import natsorted

import habitat_sim
import utils
import utils_sim_traj as ust

import nav_parser
from ContinuousControl import ContinuousControl as CC
import auto_agent as AA

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

teleop = False
display = False
stepthrough = False # if teleop, stepthrough is ignored
mapping = False # set to True to create an automatic mapping run trajectory
autoAgentName = 'RoboHop' # 'GNM' or 'RoboHop'
mapPath = "./out/maps/multiPointTrajs/learner_train/"
# get args
# args = nav_parser.parse_args("-ds hm3d -v 0.05 -c 'ceiling' -d".split(" "))
args = nav_parser.parse_args()
seed = args.seed
maxSteps = args.max_steps
print(f"Dataset: {args.dataset}, Running with seed {seed} and maxSteps {maxSteps}")

mapNames = natsorted(os.listdir(mapPath))
# for i, mapName in enumerate(mapNames):
#     if i < 2:
#         continue
if 1:
    i = int(args.mapIdx)
    mapName = mapNames[i]
    print(f"{i}: {mapName}")
    updateNavMesh = False
    glbDir = "./data/hm3d_v0.2/train/"
    test_scene_name = mapName.split("_")[0]
    glbDirs = natsorted(os.listdir(glbDir))
    glbDir_test_scene = [d for d in glbDirs if test_scene_name in d][0]
    test_scene = f"{glbDir}/{glbDir_test_scene}/{test_scene_name}.basis.glb"

    RH = AA.Agent_RoboHop(imgDir=f"{mapPath}/{mapName}/images/", modelsPath=f"./models/segment-anything/",h5FullPath=f"./out/RoboHop/learner_train/{mapName}/nodes.h5",forceRecomputeGraph=False,args=args,**{"intraNbrsAll":False, "comp_G23":False, "controlLogsDir":""})
