 # %%
import os, sys
# Compute the absolute path to the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.insert(0, parent_dir)

import habitat_sim
import utils
import utils_sim_traj as ust


test_scene = f"{parent_dir}/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
# get basename of test_scene
test_scene_name = test_scene.split("/")[-1].split(".")[0]

 # %%

sim, agent, action_names = utils.get_sim_agent(test_scene,updateNavMesh=True, agent_radius=0.5)

# BUG: island_index=1 for skokloster if recompute_navmesh is ever called (even with the default params)
island_index = sim.pathfinder.num_islands-1
path, agent_states = ust.get_trajectory_multipoint_FPS(sim,n=10,island_index=island_index)

# TODO: this plots several figures at once if display=True
folderPath = utils.createTimestampedFolderPath("./out/tmp/",test_scene_name)
_ = ust.get_images_from_agent_states(sim, agent, agent_states, display=False, saveDir=folderPath)
# %%
