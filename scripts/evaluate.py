from tqdm import tqdm
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def plot_distance_histogram(distance: np.ndarray, status: np.ndarray, save_path: Path,
                            show: bool = True, save: bool = False):
    fig, ax = plt.subplots(2)
    bins = np.arange(0.0, 0.5, 0.005)
    hist, bin_edges = np.histogram(distance[status[:, 0]], bins=bins)
    ax[0].stairs(hist, bin_edges)
    ax[0].set_title('success')
    bins = np.arange(0.5, 14, 0.5)
    hist, bin_edges = np.histogram(distance[status[:, 1]], bins=bins)
    ax[1].stairs(hist, bin_edges)
    ax[1].set_title('exceeded_steps')
    ax[1].set_xticks(bins)
    ax[1].tick_params(labelrotation=90)
    plt.suptitle('Distance histograms')
    plt.tight_layout()
    if save:
        plt.savefig(str(save_path / 'distance_histograms.png'))
    if show:
        plt.show()
    plt.close()


def plot_collision_histogram(collision_percent: np.ndarray, save_path: Path, show: bool = True, save: bool = False):
    fig, ax = plt.subplots(1)
    bins = np.arange(0.0, 1.0, 0.025)
    hist, bin_edges = np.histogram(collision_percent, bins=bins)
    ax.stairs(hist, bin_edges)
    plt.suptitle('Percent collisions histograms')
    plt.tight_layout()
    if save:
        plt.savefig(str(save_path / 'collisions_histograms.png'))
    if show:
        plt.show()
    plt.close()


def run(args):
    p = Path(args.path_results)
    globbed_txt = list(p.glob('*/*.txt'))
    globbed_csv = list(p.glob('*/*.csv',))
    bad_episodes = [
        'results',
        'CrMo8WxCyVb_0000016_chair_411_',
        'LT9Jq6dN3Ea_0000000_tv_monitor_21_',
        'Nfvxx8J5NCo_0000009_bed_267_',
        'k1cupFYWXJ6_0000000_sofa_19_',
        'k1cupFYWXJ6_0000008_chair_594_'
    ]
    num_samples = len(globbed_txt) -  len(bad_episodes) + 1
    status_codes = ['success', 'exceeded_steps']
    steps = np.ones((num_samples - 1, 1), int) * 1000
    distance_end = np.ones((num_samples - 1, 1), float) * 1000
    distance_closest = np.ones((num_samples - 1, 1), float) * 1000
    step_closest = np.ones((num_samples - 1, 1), int)
    collided_percent = np.zeros((num_samples - 1, 1), float)
    status = np.empty((num_samples - 1, 2), bool)
    stat_strs = []
    episodes = []
    i = 0
    for (path_meta, path_csv) in tqdm(zip(globbed_txt, globbed_csv)):
        try:
            if not (path_meta.stem.split('__')[0] + '_') in bad_episodes:
                with open(str(path_meta), 'r') as f:
                    meta = f.readlines()
                    step_str = meta[-2].strip('\n').split('=')[1]
                    steps[i] = int(step_str) if step_str != 'nan' else np.nan
                    if steps[i] > 500:
                        print(path_meta)
                    distance_end_str = meta[-3].strip('\n').split('=')[1]
                    distance_end[i] = float(distance_end_str) if distance_end_str != 'nan' else np.nan
                    stat = meta[-4].strip('\n').split('=')[1]
                    stat_strs.append(stat)
                    status[i] = np.equal(np.repeat(stat, 2), status_codes)
                    episodes.append(path_meta.stem)
                records = np.genfromtxt(str(path_csv), delimiter=',')[1:, :]  # remove first row
                distance_closest[i] = records[:, 5].min()
                collided_percent[i] = records[:, 9].sum() / records.shape[0]
                i += 1
        except IndexError:
            pass
    bad_sample_mask = (status == [0, 0]).all(1)
    good_sample_mask = np.logical_not(bad_sample_mask)

    good_status = status[good_sample_mask]
    bad_status = status[bad_sample_mask]

    distance_end = distance_end[good_sample_mask]
    steps = steps[good_sample_mask]
    distance_closest = distance_closest[good_sample_mask]
    step_closest = step_closest[good_sample_mask]
    collided_percent = collided_percent[good_sample_mask]

    num_good_samples = good_status.shape[0]
    results_status = (good_status.sum(0) / num_good_samples) * 100
    result_string = (
        f'total num samples: {good_status.shape[0]}\n'
        f'{status_codes[0]}={results_status[0]:.2f}, '
        f'mean_dist={distance_end[good_status[:, 0]].mean():2f}, '
        f'mean_steps={steps[good_status[:, 0]].mean()}\n'
        f'{status_codes[1]}={results_status[1]:.2f}, '
        f'mean_dist={distance_end[good_status[:, 1]].mean():2f}, '
        f'mean_steps={steps[good_status[:, 1]].mean()}\n'
        f'distance_ends <= 1.0m = {(((distance_end <= 1).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_ends <= 1.5m = {(((distance_end <= 1.5).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_ends <= 2.0m = {(((distance_end <= 2).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_ends <= 2.5m = {(((distance_end <= 2.5).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_ends <= 3.0m = {(((distance_end <= 3.0).sum() / num_good_samples) * 100):.2f}\n'
        f'\nthe closest point a run gets:\n'
        f'step_closest <= 100m = {(((step_closest <= 100).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 150m = {(((step_closest <= 150).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 200m = {(((step_closest <= 200).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 250m = {(((step_closest <= 250).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 300m = {(((step_closest <= 300).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 350m = {(((step_closest <= 350).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 400m = {(((step_closest <= 400).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 450m = {(((step_closest <= 450).sum() / num_good_samples) * 100):.2f}\n'
        f'step_closest <= 500m = {(((step_closest <= 500).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 1.0m = {(((distance_closest <= 1).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 1.5m = {(((distance_closest <= 1.5).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 2.0m = {(((distance_closest <= 2).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 2.5m = {(((distance_closest <= 2.5).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 3.0m = {(((distance_closest <= 3.0).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 3.5m = {(((distance_closest <= 3.5).sum() / num_good_samples) * 100):.2f}\n'
        f'distance_closest <= 4.0m = {(((distance_closest <= 4.0).sum() / num_good_samples) * 100):.2f}\n'
        f'\ncollisions\n'
        f'collided_percent <= 5 = {(((collided_percent <= .05).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 10 = {(((collided_percent <= .1).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 20 = {(((collided_percent <= .2).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 30 = {(((collided_percent <= .3).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 40 = {(((collided_percent <= .4).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 50 = {(((collided_percent <= .5).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 60 = {(((collided_percent <= .6).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 70 = {(((collided_percent <= .7).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 80 = {(((collided_percent <= .8).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 90 = {(((collided_percent <= .9).sum() / num_good_samples) * 100):.2f}\n'
        f'collided_percent <= 100 = {(((collided_percent <= 1.).sum() / num_good_samples) * 100):.2f}\n'
    )
    print(result_string)

    path_summary_results = p / 'summary'
    path_summary_results.mkdir(parents=True, exist_ok=True)

    episodes = np.array(episodes)
    good_episodes = episodes[good_sample_mask]
    if bad_sample_mask.sum() > 0:
        bad_status_string = np.array(stat_strs)[bad_sample_mask]
        bad_episodes = episodes[bad_sample_mask]
        print(f'Some samples were no good!')
        print(bad_episodes)
        print(bad_status_string)
        with open(str(path_summary_results / 'no_good.csv'), 'w') as f:
            for no_good, bad_stat in zip(bad_episodes, bad_status_string):
                f.write(f'{no_good}, {bad_stat}\n')

    with open(str(path_summary_results / 'results.txt'), 'w') as f:
        f.write(result_string)
    with open(str(path_summary_results / 'winners.csv'), 'w') as f:
        for res in good_episodes[good_status[:, 0]]:
            f.write(f'{res}\n')
    with open(str(path_summary_results / 'failures.csv'), 'w') as f:
        for res in good_episodes[good_status[:, 1]]:
            f.write(f'{res}\n')
    plot_distance_histogram(
        distance=distance_end, status=good_status, save_path=path_summary_results, show=args.plot_show,
        save=args.plot_save
    )
    plot_collision_histogram(
        collision_percent=collided_percent,
        save_path=path_summary_results,
        show=args.plot_show,
        save=args.plot_save
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path_results", type=str, help="Path to the folder for recording results")
    parser.add_argument("--plot_show",
                        help="Show distance histogram", action='store_true')
    parser.add_argument("--plot_save",
                        help="Save distance histogram", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
