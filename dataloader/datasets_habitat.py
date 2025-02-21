# %%
import gzip
import json
from habitat.utils.geometry_utils import quaternion_from_coeff
import habitat
import habitat_sim
import os, sys, cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from natsort import natsorted
from spatialmath import SE3
from importlib import reload

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# Compute the absolute path to the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_dir = os.path.dirname(script_dir)  # Parent directory
sys.path.insert(0, parent_dir)

import utils
import utils_sim_traj as ust


# %%
def load_from_json(filePath):
    # "~/workspace/data/inputs/instance_imagenav_hm3d_v3/val_mini/content/TEEsavR23oF.json.gz"
    with gzip.open(filePath, 'rt', encoding='utf-8') as zipfile:
        data = json.load(zipfile)
    # print(json.dumps(data, indent=4))
    return data


def getDataset_instanceimagenav(dataDir, sceneName, episodeDir=None, split='val'):
    if episodeDir is None:
        episodeDir = f"instance_imagenav_hm3d_v3/{split}/content/"
    data = load_from_json(f'{dataDir}/{episodeDir}/{sceneName}.json.gz')
    iin = habitat.datasets.image_nav.instance_image_nav_dataset.InstanceImageNavDatasetV1()
    iin.from_json(json.dumps(data))
    test_scene = f"{dataDir}/{iin.episodes[0].scene_id}"
    return iin, test_scene


def getImgFromPosRot(agent, sim, p=None, r=None, state=None, sensor=None):
    if state is None:
        state = habitat_sim.AgentState(p, r)
    agent.set_state(state)
    obs = sim.get_sensor_observations()
    if sensor is None:
        return obs
    else:
        return obs[sensor]


def getEpisode(iin, episode_id=0, agent=None, sim=None):
    e = iin.episodes[episode_id]
    # igoals = e.goals[0].image_goals
    # ig_maxFC = np.argmax([ig.frame_coverage for ig in igoals])
    # g = igoals[ig_maxFC]
    # state_g = habitat_sim.AgentState(g.position,quaternion_from_coeff(g.rotation))
    igoalviews = e.goals[0].view_points
    igv_maxIOU = np.argmax([igv.iou for igv in igoalviews])
    v = igoalviews[igv_maxIOU]
    state_g = v.agent_state
    state_s = habitat_sim.AgentState(e.start_position, quaternion_from_coeff(e.start_rotation))
    obs_s, obs_g = None, None
    if agent is not None and sim is not None:
        obs_s = getImgFromPosRot(agent, sim, state=state_s)
        obs_g = getImgFromPosRot(agent, sim, state=state_g)
    return state_s, state_g, obs_s, obs_g, e


def getSimAgent(test_scene, **kwargs):
    sim_settings = utils.get_sim_settings(scene=test_scene,
                                          width=kwargs.get('width', 640), height=kwargs.get('height', 480),
                                          hfov=kwargs.get('hfov', 58), sensor_height=1.31)
    cfg = utils.make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])
    return agent, sim


def getTrajectory(sim, start, goal, agent=None, quat_s=None, quat_g=None):
    _, pathShortest = ust.find_shortest_path(sim, start.position, goal.position)
    pathShortestInterp = ust.interpolate_path_points(pathShortest, 0.2)
    agent_states = ust.interpolate_orientation(pathShortestInterp, np.pi / 12, quat_s, quat_g)
    # ust.display_images_along_path(sim,agent,agent_states)
    # _ = ust.get_images_from_agent_states(sim,agent,agent_states,False)
    t_lin = ust.get_tgt_lin(agent_states, pathShortest)
    t_rot = ust.get_tgt_rot(agent_states, pathShortestInterp)
    action_labels = list(zip(t_lin, t_rot))
    return agent_states, pathShortestInterp, action_labels


def storeObservations2play(sim, agent, agent_states, saveDir):
    instaIdx2catIdx = utils.get_instance_to_category_mapping(sim.semantic_scene)
    catIdx2catName = utils.get_category_index_to_name_mapping(sim.semantic_scene)
    K = utils.getK_fromAgent(agent)
    T_BC = SE3.Rx(np.pi).A  # camera to base
    for i in range(len(agent_states)):
        obs_s = getImgFromPosRot(agent, sim, state=agent_states[i])
        rgb, dep, sem = obs_s['color_sensor'][:, :, :3][:, :, ::-1], obs_s['depth_sensor'], obs_s['semantic_sensor']
        # cv2.imwrite(f"{saveDir}/rgb_{i:03d}.png",rgb)
        # np.save(f"{saveDir}/dep_{i:03d}.npy",dep)
        # cv2.imwrite(f"{saveDir}/sem_{i:03d}.png",sem.astype(float))
        # # store pose (tran and rot) and both sem idx maps in a single json
        # with open(f"{saveDir}/pose_sem_{i:03d}.json", 'w') as f:
        #     json.dump({"t":agent_states[i].position.tolist(),"R":ust.hs_quat_to_array(agent_states[i].rotation).tolist(),"insta2cat":instaIdx2catIdx.tolist(),"cat2name":catIdx2catName.tolist(), "K":K}, f)
        R = ust.hs_quat_to_array(agent_states[i].rotation)
        t = agent_states[i].position
        T_WB = np.eye(4)
        T_WB[:3, :3] = R
        T_WB[:3, -1] = t.flatten()
        T_WC = T_WB @ T_BC
        np.savez(f"{saveDir}/{i:05d}.npz", **{'rgb': rgb, 'dep': dep, 'sem': sem, 't': T_WC[:3, -1], 'R': T_WC[:3, :3],
                                              'insta2cat': instaIdx2catIdx, 'cat2name': catIdx2catName, 'K': K})


def create_map_trajectory(dataDir, sceneName, eid=0, outDir="./out/maps/multiPointTrajs/", split='val'):
    iin, test_scene = getDataset_instanceimagenav(dataDir, sceneName, split=split)
    agent, sim = getSimAgent(test_scene, **{"width": 256, "height": 256, "hfov": 90})
    iigs = habitat.tasks.nav.instance_image_nav_task.InstanceImageGoalSensor(sim, None, iin)
    # goalImg = iigs.get_observation(episode=iin.episodes[eid])
    state_s, state_g, obs_s, obs_g, episode = getEpisode(iin, eid, agent, sim)
    agent_states, pathShortestInterp, action_labels = getTrajectory(sim, state_s, state_g, None,
                                                                    quat_s=state_s.rotation, quat_g=state_g.rotation)
    # storeObservations2play(sim,agent,agent_states,"./out/data2play"); exit()
    folderPath, subfolderPath_rgb, subfolderPath_depth = utils.createTimestampedFolderPath(outDir,
                                                                                           f"{sceneName}_{episode.object_category}_{episode.goal_object_id}",
                                                                                           subfolder=["images",
                                                                                                      "images_depth"])
    _ = ust.get_images_from_agent_states(sim, agent, agent_states, display=False, saveDir=subfolderPath_rgb,
                                         inc_depth=True, saveDir_depth=subfolderPath_depth)
    topdownmap = ust.display_trajectory(sim, pathShortestInterp, savePath=f"{folderPath}/map.png")
    np.save(f'{folderPath}/path.npy', pathShortestInterp)
    np.save(f'{folderPath}/agent_states.npy', agent_states)
    np.save(f'{folderPath}/episode.npy', episode)
    np.save(f'{folderPath}/obs_g.npy', obs_g)
    np.save(f'{folderPath}/action_labels.npy', action_labels)
    # save goal img
    goalImg = obs_g['color_sensor'][:, :, :3]
    cv2.imwrite(f"{folderPath}/goalImg.png", goalImg[:, :, ::-1])
    plt.imshow(obs_g['semantic_sensor'])
    # save only the image part from the plot, no axes
    plt.axis('off')
    plt.savefig(f"{folderPath}/goalImg_semantic.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def create_vpr_dataset(dataDir, outDir="./out/dataset/", split='val'):
    gt = []
    sceneNames = os.listdir(f"{dataDir}/instance_imagenav_hm3d_v3/{split}/content/")
    episodeDir = f"instance_imagenav_hm3d_v3/{split}/content/"
    for sceneName in sceneNames:
        iin, test_scene = getDataset_instanceimagenav(dataDir, sceneName[:-8], episodeDir, split)
        agent, sim = getSimAgent(test_scene)
        iigs = habitat.tasks.nav.instance_image_nav_task.InstanceImageGoalSensor(sim, None, iin)

        for gk, gv in iin.goals.items():
            print(f"Num img goals for {gk}: {len(gv.image_goals)}")
            print(f"Num view points for {gk}: {len(gv.view_points)}")
            fc_sortedInds = np.argsort([ig.frame_coverage for ig in gv.image_goals])[:5]
            oc_sortedInds = np.argsort([-ig.object_coverage for ig in gv.image_goals])[:5]
            inds = np.unique(np.concatenate((fc_sortedInds, oc_sortedInds)))[:5]

            refName = None
            for i, ind in enumerate(inds):
                img = iigs._get_instance_image_goal(gv.image_goals[ind])
                name = f"{gk}_{ind:03d}_{gv.object_category}.png"
                if i == 0:
                    dname = "ref"
                    refName = name
                else:
                    dname = "qry"
                    gt.append([name, refName, gk, ind, gv.object_category])
                print(f"{outDir}/{dname}/{name}")
                cv2.imwrite(f"{outDir}/{dname}/{name}", img[:, :, ::-1])
        sim.close()
    np.savetxt(f"{outDir}/gt.txt", gt, fmt="%s")


# %%
# example to run this file in a loop over indices as bash script
# for i in {0..16}; do python tmp.py $i; done

# main func
if __name__ == "__main__":
    split = 'train'
    idx = int(sys.argv[1])
    sceneNames = natsorted(os.listdir(f"./data/instance_imagenav_hm3d_v3/{split}/content/"))
    # dataDir,sceneName,eid = "../data/","TEEsavR23oF",6
    # create_map_trajectory("./data/","TEEsavR23oF",6)
    # for idx in tqdm(range(len(sceneNames))):
    #     if idx < 17:
    #         continue
    print(idx, sceneNames[idx][:-8])
    create_map_trajectory("./data/", sceneNames[idx][:-8], 0, outDir=f"./out/maps/multiPointTrajs/learner_{split}/",
                          split=split)
    # create_vpr_dataset("~/workspace/data/inputs/")
    # dataDir,sceneName = "~/workspace/data/inputs/","TEEsavR23oF"
    # iin, test_scene = getDataset_instanceimagenav(dataDir,sceneName)
    # agent,sim = getSimAgent(test_scene)
    # state_s,state_g, img_s, img_g = getEpisode(iin,episode_id=0,agent=agent,sim=sim)
    # agent_states, pathShortestInterp, action_labels = getTrajectory(sim,state_s,state_g,agent=None)
    # print(pathShortestInterp, agent_states)
