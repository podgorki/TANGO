 
# %%
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

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

teleop = False
display = False
if teleop: display = True
stepthrough = False # if teleop, stepthrough is ignored
mapping = False # set to True to create an automatic mapping run trajectory
autoAgentName = 'GNM' # 'GNM' or 'RoboHop'
mapPath = "./out/maps/multiPointTrajs/"
# get args
args = nav_parser.parse_args("-ds hm3d -v 0.05 -c 'ceiling' -d".split(" "))
# args = nav_parser.parse_args()
seed = args.seed
maxSteps = args.max_steps
print(f"Dataset: {args.dataset}, Running with seed {seed} and maxSteps {maxSteps}")
randColors = np.random.randint(0, 255, (200, 3))

if args.dataset == 'skokloster':
    updateNavMesh = True
    test_scene = "./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    mapName = "skokloster-castle_20240123162237032019"
    test_scene_name = test_scene.split("/")[-1].split(".")[0]
elif args.dataset == 'hm3d':
    updateNavMesh = False
    glbDir = "./data/hm3d_v0.2/val/"
    test_scene_name = "5cdEh9F2hJL"
    glbDirs = natsorted(os.listdir(glbDir))
    glbDir_test_scene = [d for d in glbDirs if test_scene_name in d][0]
    test_scene = f"{glbDir}/{glbDir_test_scene}/{test_scene_name}.basis.glb"
    # mapName = "TEEsavR23oF_sofa_132_20240408202325700113" # has depth
    # mapName = "4ok3usBNeis_chair_8_20240415155542210375"
    mapName = "5cdEh9F2hJL_toilet_36_20240430160015093231"

if args.path_rerun is None:
    outDir, outImgDir, outStateDir, controlLogsDir = utils.createTimestampedFolderPath(f"./out/runs/dump/",mapName,subfolder=["images","states","controlLogs"])
else:
    outDir, outImgDir, outStateDir, controlLogsDir = args.path_rerun, f"{args.path_rerun}/images", f"{args.path_rerun}/states", f"{args.path_rerun}/controlLogs"
    print(f"Rerunning from {args.path_rerun}")

if not teleop:
    if autoAgentName == 'GNM':
        from auto_agent import Agent_GNM
        parent_dir = "."
        autoAgent = Agent_GNM(modelconfigpath=f"{parent_dir}/auto_agents/gnm/config/models.yaml",modelweightspath=f"{parent_dir}/auto_agents/gnm/weights/",modelname="gnm_large",mapdir=f"{parent_dir}/{mapPath}/{mapName}/images/",precomputedDir=f"{parent_dir}/{mapPath}/{mapName}/GNM/")
    elif autoAgentName == 'RoboHop':
        import auto_agent as AA
        RH = AA.Agent_RoboHop(imgDir=f"{mapPath}/{mapName}/images/", modelsPath=f"./models/segment-anything/",h5FullPath=f"./out/tmp/{mapName}_nodes.h5",forceRecomputeGraph=False,args=args,**{"intraNbrsAll":False, "comp_G23":False, "controlLogsDir":controlLogsDir})
        autoAgent = RH

 # %%

sim, agent, action_names = utils.get_sim_agent(test_scene,updateNavMesh)
# action = "turn_right"
# utils.navigateAndSee(action, action_names, sim, display=True)

# create continuous control object for autoagents
cc = CC(frame_skip=1)
time_step = 1.0/cc.control_frequency

if teleop:
    pass
elif autoAgentName == 'GNM':
    agent_params = {"max_v": 0.05 , "max_w": 0.1} # m/s, # rad/s
    agent_states = np.load(f"{mapPath}/{mapName}/agent_states.npy", allow_pickle=True)#[:-6]
    agent.set_state(agent_states[0])
else:
    agent_params = {"max_v": 0.2 , "max_w": 0.6} # m/s, # rad/s
    # setup agent for specific exp
    agent_states = np.load(f"{mapPath}/{mapName}/agent_states.npy", allow_pickle=True)#[:-6]
    if args.dataset == 'skokloster':
        # RH.init_episode(minTrajLength=9,goalNode=651,startNode=392)
        RH.init_episode(minTrajLength=10,seed=seed)
    elif args.dataset == 'hm3d':
        goalImgIdx = len(agent_states)-1
        episode = np.load(f"{mapPath}/{mapName}/episode.npy", allow_pickle=True)[()]
        obs_g = np.load(f"{mapPath}/{mapName}/obs_g.npy", allow_pickle=True)[()]
        goalMaskBinary = obs_g['semantic_sensor']==int(episode.goal_object_id)
        goalMaskBinary = cv2.resize(goalMaskBinary.astype(float), (RH.img_w, RH.img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mapNodeInds_in_goalImg = np.argwhere(RH.nodeID_to_imgRegionIdx[:,0] == goalImgIdx).flatten()
        mapMasks = RH.nodes2key(mapNodeInds_in_goalImg,'segmentation').transpose(1,2,0)
        mask_and = np.logical_and(mapMasks, goalMaskBinary[:,:,None])
        mask_or = np.logical_or(mapMasks, goalMaskBinary[:,:,None])
        iou = mask_and.sum(0).sum(0) / mask_or.sum(0).sum(0)
        goalNode = mapNodeInds_in_goalImg[iou.argmax()]
        RH.init_episode(minTrajLength=len(agent_states)-2,goalImgIdx=goalImgIdx,goalNode=goalNode)
    if args.resume_episode is None:
        agent.set_state(agent_states[RH.localizedImgIdx])
    else:
        episodePast = np.load(f'{args.resume_episode}/episode.npy', allow_pickle=True)[()]
        RH.localizedImgIdx = episodePast['localizedImgIdx']
        agent.set_state(episodePast['final_state'])
    if args.weight_string is not None:
        RH.mapNodeWeightStr = args.weight_string

    if args.feed_states:
        maxSteps = len(agent_states) - 1
        print(f"Feeding states, maxSteps set to {maxSteps}")

if mapping:
    folderPath, subfolderPath = utils.createTimestampedFolderPath(mapPath,test_scene_name,subfolder="images")
    ust.create_map_trajectory(sim,agent,folderPath,subfolderPath,numPoints=10)

# %%
step = -1
rgb_img = None
depth = None
complete = False
plt.figure(figsize=(12, 8))
display_img = None
# Main loop
try:
    while step < maxSteps:
        step += 1
        # Get keyboard input
        # key_command = input("Enter command (w/a/d): ").lower()
        if step == 0:
            key_command = 'w'
            autoaction = [0,0]
            action = utils.map_keyB2Act(key_command)
            v,w = 0., 0.
        else:
            # read from keyboard or autoagent
            if teleop or stepthrough:
                key_command = utils.get_kb_command()
                action = utils.map_keyB2Act(key_command)
                if action is None:
                    print("Invalid Key, use w/a/d")
                    continue
            # just roam around
            if teleop:
                pass
            # use an autoagent to process images and get actions (v,w)
            else:
                if autoAgentName == 'GNM':
                    v,w,dx,theta = utils.get_autoagent_action(autoAgent,rgb_img,agent_params,time_step)
                    autoaction = [dx, theta]
                elif autoAgentName == 'RoboHop':
                    v,w,x_off,hop,display_img = RH.get_control_signal(rgb_img,depth)
                    if RH.done or step == maxSteps:
                        curr_state = agent.get_state()
                        t_err, r_err = utils.compute_pose_err(curr_state,agent_states[RH.nodeID_to_imgRegionIdx[RH.goalNode][0]])
                        print(f"\nGoal reached, with t_err {t_err:.3f} and r_err {r_err:.3f}")
                        RH.save_episode(**{"t_err":t_err, "r_err":r_err, "final_state":curr_state})
                        complete = True
                        break
                    autoaction = [v,w]
                print(autoaction, v, w)

        # Execute actions and get new observations
        # pass actions from keyboard
        if teleop or stepthrough:
            observations = sim.step(action)
        # pass actions from a given set
        elif args.feed_states:
            agent.set_state(agent_states[step])
            observations = sim.get_sensor_observations()
        # pass actions obtained from autoagent (cc.act)
        else:
            # observations = cc.act(*cc.generate_random_continuous_actions(),agent,sim)[-1]
            observations = cc.act(v,w,agent,sim)[-1]

        rgb_obs = observations["color_sensor"]
        if args.use_depth:
            depth = observations["depth_sensor"]
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        # save image in folderPath, with names zero-padded to 5 digits using f-strings
        rgb_img.save(f"{outImgDir}/{step:05d}.png")
        np.save(f"{outStateDir}/{step:05d}.npy",agent.get_state())

        if display:
            if display_img is None:
                # write the autoaction ([float, float]) as text (each as 2 decimal) on image using cv2, color pastel blue
                display_img = np.array(rgb_img.convert('RGB'))
                cv2.putText(display_img,f"{autoaction[0]:.2f},{(-autoaction[1]):.2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 204, 204),2,cv2.LINE_AA)

                if "semantic_sensor" in observations:
                    semantic = observations["semantic_sensor"]
                    semantic = utils.get_instance_to_category_mapping(sim.semantic_scene)[semantic][:,:,1]
                    # map ids to colors
                    semantic = randColors[semantic]
                    display_img = np.column_stack([display_img, semantic])
                # if "depth_sensor" in observations:
                #     depth = observations["depth_sensor"]
                #     display_img = np.column_stack([display_img, depth])
            plt.imshow(display_img)
            plt.pause(0.001)  # pause a bit so that plots are updated

except KeyboardInterrupt:
    pass
finally:
    sim.close()
    if autoAgentName=='RoboHop' and not complete:
        print("Saving episode after a failed/aborted run ...")
        if not teleop:
            RH.save_episode()
    # exit clean
    if teleop or stepthrough:
        stdscr = curses.initscr()
        stdscr.keypad(0)
        curses.echo()
        curses.endwin()

# %%
