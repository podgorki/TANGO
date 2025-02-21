# %%
import os
import sys
import subprocess
from os.path import exists, join, isdir
import numpy as np
import matplotlib.pyplot as plt

# Compute the absolute path to the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.insert(0, parent_dir)

import utils

# %%
def generate_videos(path,mode='teach'):
    '''
    Generate videos from the images in the subdirectories of the given path.
    path: str
        The path to the directory containing the subdirectories with images.
    mode: str
        The mode to run the function in. Can be 'teach' or 'repeat'.
        'teach' generates a video from the images in the images directory.
        'repeat' generates a video from the images in the matchings directory.
    '''
    # List all entries in the current directory
    entries = os.listdir(path)
    print(f"Found {len(entries)} entries in {path}")
    entries = [join(path,entry) for entry in entries]
    
    # Filter out subdirectories
    subdirs = [entry for entry in entries if isdir(entry)]
    print(f"Found {len(subdirs)} subdirectories")
    
    # Iterate through each subdirectory
    for subdir in subdirs:
        if mode == 'repeat':
            subdir = join(subdir,'controlLogs')
        print(f"Processing {subdir}")
        # Change into the subdirectory
        if not exists(os.path.abspath(subdir)):
            print(f"Skipping {subdir} that doesn't exist")
            continue
        os.chdir(subdir)
        
        if mode == 'teach':
            command = 'ffmpeg -framerate 3 -i ./images/%05d.png -pix_fmt yuv420p -c:v libx264 teach.mp4'
            subprocess.run(command, shell=True)
        # Check if the matchings directory exists
        elif mode == 'repeat' and exists('matchings'):
            # Run the ffmpeg command
            command = 'ffmpeg -framerate 3 -i ./matchings/combinedImg_%05d.png -pix_fmt yuv420p -c:v libx264 repeat.mp4'
            subprocess.run(command, shell=True)

def plotStats(stats1,savePath="./out/experiments/plots/"):
    if not os.path.exists(savePath):
        os.makedirs(savePath, exist_ok=True)
    print(f"Plotting t_err and r_err distributions for {len(stats1)} successful experiments")
    plt.figure()
    plt.hist(stats1[:,3].astype(float), bins=20)
    plt.title(f't_err (completed exp = {len(stats1)})')
    plt.savefig(f'{savePath}/t_err.png')
    plt.figure()
    plt.hist(stats1[:,4].astype(float), bins=20)
    plt.title(f'r_err (completed exp = {len(stats1)})')
    # save both plots
    plt.savefig(f'{savePath}/r_err.png')
    plt.show()

def check_failure(path,maxSteps=200):
    recompute_err = True
    allPaths = sorted([join(path,p,"controlLogs","matchings") for p in os.listdir(path)])
    t_th, r_th = 5, 30
    stats = []
    for pi,p in enumerate(allPaths):
        failedExp = False
        stepRunOut = False
        t_err, r_err, expDir = 1e6, 180, ""
        success = False
        if exists(p):
            numFiles = len(os.listdir(p))
            expDir = p.split("/")[-3]
            if numFiles != maxSteps:
                pass
                # print(pi, expDir, f" has {numFiles} files")
            else:
                stepRunOut = True
            # check if episode.npy exists in controlLogs
            episodeFilePath = join(p,'..','episode.npy')
            if exists(episodeFilePath):
                e = np.load(episodeFilePath, allow_pickle=True)[()]
                if 't_err' in e:
                    t_err = e['t_err']
                    r_err = e['r_err']
                    if recompute_err:
                        mapName = "skokloster-castle_20240123162237032019"
                        assert(mapName in expDir)
                        agent_states = np.load(f"./out/maps/multiPointTrajs/{mapName}/agent_states.npy", allow_pickle=True)#[:-6]

                        finalStatePath = join(p,'..','..',"states",f"{numFiles-1:05d}.npy")
                        finalState = np.load(finalStatePath,allow_pickle=True)[()]
                        for imgIdx in range(e['goalImgIdx']-1,e['goalImgIdx']+2):
                            t_err, r_err = utils.compute_pose_err(finalState,agent_states[imgIdx])
                            if t_err < t_th and r_err < r_th:
                                success = True
                                break
                            # print(f"Episode: {pi}: t_err: {t_err}, r_err: {r_err}")
                else:
                    failedExp = True
                    # assert(False) # TODO
            else:
                # print("episode.npy doesn't exist")
                failedExp = True
        else:
            failedExp = True
        stats.append([pi, int(failedExp), int(stepRunOut), t_err, r_err, expDir, int(success)])
    stats = np.array(stats)
    failedMask = stats[:,1].astype(int)==1
    successes = stats[:,6].astype(int)==1
    print(stats[successes,0])
    indsFailed, indsRan = np.where(failedMask)[0], np.where(~failedMask)[0]
    stats1 = stats[indsRan]
    # plotStats(stats1,savePath=f"./out/experiments/plots/{path.split('/')[-1]}/stat1/")
    indsMaxSteps = np.where(stats1[:,2].astype(int)==1)[0]
    indsSelfExit = np.where(np.logical_and(~stats[:,2].astype(int).astype(bool), ~stats[:,1].astype(int).astype(bool)))[0]
    print(f"Total: {len(stats)}, Failed: {len(indsFailed)}, Ran: {len(indsRan)}, MaxSteps: {len(indsMaxSteps)} Success: {successes.sum()}")
    stats2 = stats[indsSelfExit]
    # plotStats(stats2,savePath=f"./out/experiments/plots/{path.split('/')[-1]}/stat2/")
    # print("Sorted Stats by t_err")
    # print(stats2[np.argsort(stats2[:,-3])])
    return {"indsSelfExit":indsSelfExit, "indsFailed":indsFailed, "indsRan":indsRan, "indsMaxSteps":indsMaxSteps, "stats1":stats1, "stats2":stats2,  "stats":stats}

# %%
if __name__ == '__main__':
    generate_videos(sys.argv[1])
    # check_failure(sys.argv[1])
