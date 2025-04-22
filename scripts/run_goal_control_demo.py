import os
import yaml
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from src.logger import default_logger
from src.experiments import task_setup
from src.utils import split_observations, get_traversibility

if 'LOG_LEVEL' not in os.environ:
    os.environ['LOG_LEVEL'] = "DEBUG"
default_logger.setup_logging(level=os.environ['LOG_LEVEL'])

torch.backends.cudnn.benchmark = True


def run(args):
    # set up all the paths
    path_dataset = Path(args.path_dataset)
    path_scenes_root_hm3d = path_dataset / 'hm3d_v0.2' / args.split
    path_episode_root = path_dataset / f'hm3d_iin_{args.split}'
    print(f'Root path for episodes: {path_episode_root}')

    path_results_folder = task_setup.init_results_dir_and_save_cfg(
        args, default_logger
    )
    print("\nConfig file saved in the results folder!\n")

    preload_data = task_setup.preload_models(args)

    episodes = task_setup.load_run_list(args, path_episode_root)[args.start_index:args.end_index:args.steps_index]

    for ei, path_episode in tqdm(
            enumerate(episodes), total=len(episodes), desc=f'Processing Episodes (Total: {len(episodes)})'
    ):
        episode_name = path_episode.parts[-1].split('_')[0]
        path_scene_hm3d = sorted(path_scenes_root_hm3d.glob(f'*{episode_name}'))

        if len(path_scene_hm3d) > 0:
            scene_name_hm3d = str(sorted(path_scene_hm3d[0].glob('*basis.glb'))[0])

            episode_runner = task_setup.Episode(
                args,
                path_episode,
                scene_name_hm3d,
                path_results_folder,
                preload_data
            )
            print("FUCK YOU")
            if args.plot:
                ax, plt = episode_runner.init_plotting()

            for step in range(args.max_steps):
                if episode_runner.is_done():
                    break

                observations = episode_runner.sim.get_sensor_observations()
                display_img, depth, semantic_instance_sim = split_observations(observations)

                if args.infer_depth:
                    depth_scale = 0.44
                    depth = preload_data["depth_model"].infer(display_img) * depth_scale

                episode_runner.get_goal(display_img, depth, semantic_instance_sim)

                if not args.infer_traversable:  # override the FastSAM traversable mask
                    episode_runner.traversable_mask = get_traversibility(
                        torch.from_numpy(semantic_instance_sim),
                        episode_runner.traversable_class_indices
                    ).numpy()

                episode_runner.get_control_signal(depth)
                episode_runner.execute_action()

                if args.plot:
                    episode_runner.plot(ax, plt, step, display_img, depth, semantic_instance_sim)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", "-c",
                        help="Path to the config file", default="../configs/tango.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_file = args.config_file
    print(f'Running from config file: {config_file}')

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Config File {config_file} params: {config}")
            # pass the config to the args
            for k, v in config.items():
                setattr(args, k, v)

    # setup traversable classes for TANGO
    setattr(args, "traversable_class_names", [
        'floor', 'flooring', 'floor mat', 'floor vent', 'carpet', 'mat', 'rug', 'doormat', 'shower floor', 'pavement',
        'ground', 'tiles'
    ])
    run(args)
