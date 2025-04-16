import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import curses
import datetime
import os
import quaternion
import cv2
import habitat_sim


def dict_to_args(cfg_dict):
    args = type('', (), {})()
    for k, v in cfg_dict.items():
        setattr(args, k, v)
    return args


def get_default_args():
    args_dict = {
        'method': 'tango',
        'goal_source': 'gt_metric',
        'graph_filename': None,
        'max_start_distance': 'easy',
        'threshold_goal_distance': 0.5,
        'debug': False,
        'max_steps': 500,
        'run_list': '',
        'path_run': '',
        'path_models': None,
        'log_robot': True,
        'save_vis': False,
        'plot': False,
        'infer_depth': False,
        'infer_traversable': False,
        'segmentor': 'fast_sam',
        'task_type': 'original',
        'use_gt_localization': False,
        'env': 'sim',
        'goal_gen': {
            'text_labels': [],
            'matcher_name': 'lightglue',
            'map_matcher_name': 'lightglue',
            'geometric_verification': True,
            'match_area': False,
            'goalNodeIdx': None,
            'edge_weight_str': None,
            'use_goal_nbrs': False,
            'plan_da_nbrs': False,
            'rewrite_graph_with_allPathLengths': False,
            'loc_radius': 4,
            'do_track': False,
            'subsample_ref': 1,
        },
        'sim': {
            'width': 320,
            'height': 240,
            'hfov': 120,
            'sensor_height': 0.4
        },
    }
    # args_dict as attributes of args
    return dict_to_args(args_dict)


def get_sim_settings(scene, default_agent=0, sensor_height=1.5, width=256, height=256, hfov=90):
    sim_settings = {
        "scene": scene,  # Scene path
        "default_agent": default_agent,  # Index of the default agent
        "sensor_height": sensor_height,  # Height of sensors in meters, relative to the agent
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "hfov": hfov
    }
    return sim_settings


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    else:
        annotConfigPath = findAnnotationPath(settings["scene"])
        if annotConfigPath is not None:
            print(f"Annotation file found: {annotConfigPath}")
            sim_cfg.scene_dataset_config_file = annotConfigPath
        else:
            print(f"Annotation file not found for {settings['scene']}")

    # agent
    hardware_config = habitat_sim.agent.AgentConfiguration()
    # # Modify the attributes you need
    # hardware_config.height = 20  # Setting the height to 1.6 meters
    # hardware_config.radius = 10  # Setting the radius to 0.2 meters
    # discrete actions defined for objectnav task in habitat-lab/habitat/config/habitat/task/objectnav.yaml
    custom_action_dict = {
        'stop': habitat_sim.ActionSpec(name='move_forward', actuation=habitat_sim.ActuationSpec(amount=0))}
    for k in hardware_config.action_space.keys():
        custom_action_dict[k] = hardware_config.action_space[k]
    custom_action_dict['look_up'] = habitat_sim.ActionSpec(name='look_up',
                                                           actuation=habitat_sim.ActuationSpec(amount=30))
    custom_action_dict['look_down'] = habitat_sim.ActionSpec(name='look_down',
                                                             actuation=habitat_sim.ActuationSpec(amount=30))

    hardware_config.action_space = custom_action_dict
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    # add depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    hardware_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [hardware_config])


def get_sim_agent(test_scene, updateNavMesh=False, agent_radius=0.75, width=320, height=240, hfov=90,
                  sensor_height=1.5):
    sim_settings = get_sim_settings(scene=test_scene, width=width, height=height, hfov=hfov,
                                    sensor_height=sensor_height)
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
    sim.pathfinder.seed(42)
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

    if updateNavMesh:
        # update navmesh to avoid tight spaces
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = agent_radius
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        # sim_topdown_map = sim.pathfinder.get_topdown_view(0.1, 0)

    return sim, agent, action_names


def getK_fromParams(hfovDeg, width, height):
    hfov = np.deg2rad(float(hfovDeg))
    # K = np.array([
    # [1 / np.tan(hfov / 2.), 0., 0.],
    # [0., 1 / np.tan(hfov / 2.) * (width / height), 0.],
    # [0., 0.,  1]])
    K = np.array([
        [(width / 2.) / np.tan(hfov / 2.), 0., width / 2.],
        [0., (height / 2.) / np.tan(hfov / 2.), height / 2.],
        [0., 0., 1]])
    return K


def getK_fromAgent(agent):
    specs = agent.agent_config.sensor_specifications[0]
    return getK_fromParams(specs.hfov, specs.resolution[1], specs.resolution[0])


def value_to_colour(values, vmin=None, vmax=None, cmName='jet'):
    cmapPaths = matplotlib.colormaps.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm


def visualize_flow(cords_org, cords_dst, img=None, colors=None, norm=None, weights=None, cmap='jet', colorbar=True,
                   display=True, fwdVals=None):
    diff = cords_org - cords_dst
    dpi = 100
    img_height, img_width = img.shape[:2]  # Get the image dimensions
    fig_width, fig_height = img_width / dpi, img_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if img is not None: ax.imshow(img)
    # for i in range(len(currImg_mask_coords)):
    #     ax.plot(currImg_mask_coords[i,0],currImg_mask_coords[i,1],'o',color='r')
    #     ax.plot(refNodes_mask_coords[matchInds[i],0],refNodes_mask_coords[matchInds[i],1],'o',color='b')
    if fwdVals is not None:
        # plot a diamond for negative values and a circle for positive values, size = val
        pointTypeMask = fwdVals > 0
        ax.scatter(*(cords_org[pointTypeMask].T), c=colors[pointTypeMask], s=abs(fwdVals[pointTypeMask]) * 40,
                   marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(*(cords_org[~pointTypeMask].T), c=colors[~pointTypeMask], s=abs(abs(fwdVals[~pointTypeMask])) * 40,
                   marker='X', edgecolor='white', linewidth=0.5)
    if weights is not None:
        weightedSum = (weights[:, None] * diff).sum(0)
        ax.quiver(*(np.array([160, 120]).T), weightedSum[0], weightedSum[1], color='black', edgecolor='white',
                  linewidth=0.5)
    ax.quiver(*(cords_org.T), diff[:, 0], diff[:, 1], color=colors, edgecolor='white', linewidth=0.5)
    if colorbar: add_colourbar(ax, plt, norm, cmap)
    ax.set_xlim([0, img_width])
    ax.set_ylim([img_height, 0])
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if display:
        plt.show()
    else:
        # return the figure as image (same size as img imshow-ed above)
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis = cv2.resize(vis, (img.shape[1], img.shape[0]))
        plt.close(fig)
        return vis


def add_colourbar(ax, plt, norm=None, cmap='jet'):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create a ScalarMappable object with the "autumn" colormap
    if norm is None:
        norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add a colorbar to the axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")

    # Customize the colorbar
    cbar.set_label('Colorbar Label', labelpad=10)
    cbar.ax.yaxis.set_ticks_position('right')


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    # print(np.array(rgb_img))

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=True)


def navigateAndSee(action, action_names, sim, display=False):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])


# Function to translate keyboard commands to action strings
def map_keyB2Act(key_command):
    if key_command == 'w':
        action = 'move_forward'
    elif key_command == 'a':
        action = 'turn_left'
    elif key_command == 'd':
        action = 'turn_right'
    else:
        return None
    return action


def get_kb_command():
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.keypad(1)

    key_command = stdscr.getch()
    key_mapping = {
        ord('w'): 'w',
        ord('a'): 'a',
        ord('d'): 'd',
        curses.KEY_UP: 'w',
        curses.KEY_LEFT: 'a',
        curses.KEY_RIGHT: 'd'
    }
    command = key_mapping.get(key_command)

    curses.nocbreak()
    stdscr.keypad(0)
    curses.echo()
    curses.endwin()

    return command


def apply_velocity(agent, sim, velocity, rotation_velocity, time_step=0.1):
    # Update position))
    forward_vec = habitat_sim.utils.quat_rotate_vector(agent.state.rotation, np.array([0, 0, -1.0]))
    new_position = agent.state.position + forward_vec * velocity * time_step

    # Update rotation
    new_rotation = habitat_sim.utils.quat_from_angle_axis(rotation_velocity * time_step, np.array([0, 1.0, 0]))
    new_rotation = new_rotation * agent.state.rotation

    # Set the new state
    agent.state.position = new_position
    agent.state.rotation = new_rotation
    print(agent.state)

    # Step the physics simulation
    sim.step_physics(time_step)

    observations = sim.get_sensor_observations()

    return observations


def createTimestampedFolderPath(outdir, prefix, subfolder="", excTime=False):
    """
    Create a folder with a timestamped name in the outdir
    :param outdir: where to create the folder
    :param prefix: prefix for the folder name
    :param subfolder: subfolder name, can be a list of subfolders
    :return: paths to the created folder and subfolders
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S%f')
    if excTime: formatted_time = ""
    folder_path = f'{outdir}/{prefix}_{formatted_time}'
    if type(subfolder) == str:
        subfolder = [subfolder]
    sfPaths = []
    for sf in subfolder:
        subfolder_path = f'{outdir}/{prefix}_{formatted_time}/{sf}'
        os.makedirs(subfolder_path, exist_ok=True)
        sfPaths.append(subfolder_path)
    return folder_path, *sfPaths


def get_autoagent_action(autoagent, currImg, agent_params, time_step):
    autoagent.maintain_history(currImg)
    dists, wayps = [], []
    for mapimg in autoagent.topomap:
        dist, wayp = autoagent.predict_currHistAndGoal(autoagent.currImgHistory, mapimg)
        dists.append(dist)
        wayps.append(wayp)
    ptr = np.argmin(dists)
    # autoagent.updateLocalMap(ptr)
    print(ptr, autoagent.localmapIdx)
    wayp = wayps[min(ptr + 2, len(autoagent.topomap) - 1)][0][2]
    dx, dy = wayp[:2]
    theta = np.arctan(dy / dx) / 3.14 * 180
    v, w = autoagent.waypoint_to_velocity(wayp, agent_params, time_step)
    return v, w, dx, theta


def compute_pose_err(s1, s2):
    """
    Compute the position and rotation error between two agent states
    :param s1: habitat_sim.AgentState
    :param s2: habitat_sim.AgentState
    :return: (float, float) position error, rotation error (degrees)
    """
    pos_err = np.linalg.norm(s1.position - s2.position)
    rot_err = np.rad2deg(quaternion.rotation_intrinsic_distance(s1.rotation, s2.rotation))
    return pos_err, rot_err


# Habitat Semantics
def findAnnotationPath(scenePath):
    # find split name from among ['train', 'val', 'test', 'minival']
    split = None
    for s in ['train', 'minival', 'val', 'test']:  # TODO: 'val' inside 'minival'
        if s in scenePath:
            split = s
            pathTillSplit = scenePath.split(split)[0]
            break
    if split is None:
        return None
    else:
        return f"{pathTillSplit}/{split}/hm3d_annotated_{split}_basis.scene_dataset_config.json"


def print_regions(regions, max_regions=10, max_objects=10):
    region_count = 0
    for region in regions:
        category = region.category.name() if region.category is not None else None
        print(
            f"\t Region id:{region.id}, {category=},"
            f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        )
        object_count = 0
        for obj in region.objects:
            print(
                f"\t \t Object id:{obj.id}, category:{obj.category.name()},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
            object_count += 1
            if object_count >= max_objects:
                break
        region_count += 1
        if region_count >= max_regions:
            break


def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        print_regions(level.regions, limit_output, limit_output)

    if len(scene.levels) == 0:
        print_regions(scene.regions, limit_output, limit_output)

    # # Print semantic annotation information (id, category, bounding box details)
    # # about levels, regions and objects in a hierarchical fashion
    # scene = sim.semantic_scene
    # print_scene_recur(scene)


def obj_id_to_int(obj):
    return int(obj.id.split("_")[-1])


def get_instance_to_category_mapping(semanticScene):
    instance_id_to_label_id = np.array(
        [[obj_id_to_int(obj), obj.category.index()] for obj in semanticScene.objects])
    return instance_id_to_label_id


def get_instance_index_to_name_mapping(semanticScene):
    instance_index_to_name = np.array([[i, obj.category.name()] for i, obj in enumerate(semanticScene.objects)])
    return instance_index_to_name


def get_instance_id_to_region_id_mapping(semantic_scene):
    instance_index_to_region_id = np.array(
        [[obj_id_to_int(obj), int(obj.region.id[1:])] for obj in semantic_scene.objects])

    # check if object ids iterate exactly over total objects
    assert (instance_index_to_region_id[-1, 0] == len(semantic_scene.objects) - 1)
    return instance_index_to_region_id


def get_region_id_to_instance_id_dict(semantic_scene):
    region_id_to_instance_id = {}
    for region in semantic_scene.regions:
        region_key = int(region.id[1:])
        region_id_to_instance_id[region_key] = []
        for instance in region.objects:
            instance_id = int(instance.id.split("_")[-1])
            region_id_to_instance_id[region_key].append(instance_id)
    return region_id_to_instance_id


def get_instance_id_to_all_dict(semantic_scene, save_explicit_dict=False):
    instance_id_to_all = {}
    for instance in semantic_scene.objects:
        instance_id = int(instance.id.split("_")[-1])
        if save_explicit_dict:
            instance = {
                "category_name": instance.category.name(),
                "category_index": instance.category.index(),
                "id": instance.id,
                "semantic_id": instance.semantic_id,
                "obb_center": instance.obb.center,
                "obb_sizes": instance.obb.sizes,
                "obb_rotation": instance.obb.rotation,
                "obb_world_to_local": instance.obb.world_to_local,
                "obb_local_to_world": instance.obb.local_to_world,
            }
        instance_id_to_all[instance_id] = instance
    return instance_id_to_all


def sample_goal_instances_across_regions(semantic_scene, seed=None):
    if seed is not None:
        np.random.seed(seed)

    cat_to_avoid = ['Unknown', 'wall', 'ceiling', 'floor']
    goal_instance_ids = []
    goal_instance_coords = []
    print("Num regions:", len(semantic_scene.regions))
    for region in semantic_scene.regions:

        # sample an instance not in cat_to_avoid
        instances_filtered = [insta for insta in region.objects if insta.category.name() not in cat_to_avoid]
        if len(instances_filtered) == 0:
            continue
        instance = np.random.choice(instances_filtered, replace=False)
        instance_coords = instance.aabb.center
        goal_instance_ids.append(obj_id_to_int(instance))
        goal_instance_coords.append(instance_coords)

        print(
            f"Region: {region.id}, Instance: {goal_instance_ids[-1]}, Category: {instance.category.name()}, coords: {instance_coords}, region center: {region.aabb.center}")

    return goal_instance_ids, goal_instance_coords


def sample_goal_instances_across_regions_indirect(semantic_scene, num_goals=2, repeat_regions=False):
    cat_to_avoid = ['Unknown', 'wall', 'ceiling', 'floor']
    reg_to_insta_dict = get_region_id_to_instance_id_dict(semantic_scene)
    insta_to_cat_map = get_instance_index_to_name_mapping(semantic_scene)
    insta_to_all_dict = get_instance_id_to_all_dict(semantic_scene)

    # sample regions
    num_extra_samples = 5  # to avoid regions with no filtered instances
    reg_ids = list(reg_to_insta_dict.keys())
    num_regions_to_sample = min(num_goals + num_extra_samples, len(reg_ids))
    reg_ids = np.random.choice(reg_ids, num_regions_to_sample, replace=repeat_regions)

    goal_instance_ids = []
    goal_instance_coords = []
    i = -1
    while len(goal_instance_ids) < num_goals:
        i += 1
        reg_id = reg_ids[i]

        # sample an instance not in cat_to_avoid
        insta_ids = reg_to_insta_dict[reg_id]
        insta_ids_filtered = [insta_id for insta_id in insta_ids if insta_to_cat_map[insta_id][1] not in cat_to_avoid]
        if len(insta_ids_filtered) == 0:
            continue
        insta_id = np.random.choice(insta_ids_filtered)
        insta_coords = insta_to_all_dict[insta_id].aabb.center
        goal_instance_ids.append(insta_id)
        goal_instance_coords.append(insta_coords)

        print(
            f"Region: {reg_id}, Instance: {insta_id}, Category: {insta_to_cat_map[insta_id][1]}, coords: {insta_coords}, region center: {semantic_scene.regions[reg_id].aabb.center}")

    return goal_instance_ids, goal_instance_coords


def obs_from_state(episode, state, sensor="color_sensor"):
    episode.agent.set_state(state)
    observations = episode.sim.get_sensor_observations()
    if sensor == "color_sensor":
        obs = np.array(Image.fromarray(observations["color_sensor"], mode="RGBA").convert('RGB'))
    else:
        obs = observations[sensor]
    return obs


def getImg(sim):
    observations = sim.get_sensor_observations()
    rgb = observations["color_sensor"]
    depth = observations["depth_sensor"]
    semantic = None
    if "semantic_sensor" in observations:
        semantic = observations["semantic_sensor"]
    return rgb, depth, semantic


def get_hm3d_scene_name_from_episode_path(path_episode, path_scenes_root_hm3d):
    episode_name = path_episode.parts[-1].split('_')[0]
    path_scene_hm3d = sorted(path_scenes_root_hm3d.glob(f'*{episode_name}'))[0]
    scene_name_hm3d = str(sorted(path_scene_hm3d.glob('*basis.glb'))[0])
    return scene_name_hm3d
