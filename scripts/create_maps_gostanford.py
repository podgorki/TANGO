 
# %%
# USAGE:
# bash: for i in {1751..2000}; do CUDA_VISIBLE_DEVICES=1 python -m scripts.create_maps_gostanford -i $i; done >> logs22.txt
# or
# cat out/inds2recompute_noh5.txt | head -n 250 | while read i; do python -m scripts.create_maps_gostanford -i $i; done >> logs1.txt
# or
# sed -n '250,500p' out/inds2recompute_noh5.txt | while read i; do python -m scripts.create_maps_gostanford -i $i; done >> logs2.txt
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import curses
from importlib import reload
import os
from os.path import expanduser
import sys
from natsort import natsorted

import nav_parser
import auto_agent as AA

# get args
# args = nav_parser.parse_args("-ds hm3d -v 0.05 -c 'ceiling' -d".split(" "))
args = nav_parser.parse_args()
seed = args.seed
args.clip_csv = ""

dName = "go_stanford"
mapPath = f"{expanduser('~')}/fastdata/navigation/{dName}/"
mapNames = natsorted(os.listdir(mapPath))

mode = "mapIdx" # from "loop", "mapIdx"

if mode == "loop":
    inds2recompute = np.load("./out/inds2recompute_noh5.npy")
    print(f"inds2recompute: {inds2recompute}")
else:
    inds2recompute = [int(args.mapIdx)]

for i in inds2recompute:
    mapName = mapNames[i]
    print(f"{i}: {mapName}")

    sam_kwargs = {'points_per_side': 32, 'pred_iou_thresh': 0.86, 'stability_score_thresh': 0.92, 'crop_n_layers': 0, 'crop_n_points_downscale_factor': 0, 'min_mask_region_area': 20}
    kwargs = {
        "intraNbrsAll":False, "comp_G23":False, "comp_G3": True, "controlLogsDir":"",
        "img_eidx": -1, "desired_width": 160, "desired_height": 120, "detect": None, "segment": 'sam', 'compressMask': True, 'sam_kwargs': sam_kwargs, 'forceRecomputeH5': True
        }
    RH = AA.Agent_RoboHop(imgDir=f"{mapPath}/{mapName}/", modelsPath=f"./models/segment-anything/",h5FullPath=f"./out/RoboHop/{dName}/{mapName}/nodes.h5",forceRecomputeGraph=False,args=args,**kwargs)
