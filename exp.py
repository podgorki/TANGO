# run teleop.py with different seed values 0 to 100
# store output logs in a dir per experiment with zero padding
# wait for each experiment to finish before starting the next

import subprocess
import os
from os.path import join, exists
import numpy as np

outDir = f"{os.path.expanduser('~')}/workspace/data/outputs/SegmentMap/dump/skokloster-050-5/"
runsPaths = sorted(os.listdir(outDir))
allPaths = sorted([join(outDir,p,"controlLogs","matchings") for p in runsPaths])
dryRun = True
mode = "rerun" # "rerun"
for i in range(100):
    if mode == "rerun":
        # check if episode.npy exists
        episodeFilePath = join(allPaths[i],'..','episode.npy')
        if exists(episodeFilePath):
            e = np.load(episodeFilePath, allow_pickle=True)
            # check t_err exists as a key
            if 't_err' in e.item().keys():
                print(f"Skipping experiment {i}")
                continue
            else:
                print(f"Episode file found for experiment {i} but t_err key not found")
        else:
            print(f"Episode file not found for experiment {i}")
    
    print(f"Starting experiment {i}")

    # run teleop.py with different seed values 0 to 100
    if mode == "rerun":
        cmd = f"python teleop.py -pr {outDir}/{runsPaths[i]} -m 200 -s {i}"
    else:
        cmd = f"python teleop.py -m 200 -s {i}"

    if dryRun:
        print(cmd)
    else:
        logs = subprocess.run(cmd.split(" "), cwd="./", capture_output=True, text=True).stdout.strip("\n")

        # store output logs in a dir per experiment with zero padding
        logFile = f"./out/experiments/{str(i).zfill(3)}.txt"
        print(logs)
        with open(logFile, "w") as text_file:
            text_file.write(logs)
        print(f"Experiment {i} finished")

print("All experiments finished")