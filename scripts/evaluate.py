from tqdm import tqdm
from pathlib import Path
import numpy as np


def run():
    p = Path('')
    status_codes = ['success', 'exceeded_steps', 'no_traversable']
    steps = np.empty((len(list(p.glob('*.txt'))), 1))
    distance = np.empty((len(list(p.glob('*.txt'))), 1))
    status = np.empty((len(list(p.glob('*.txt'))), 3), bool)
    for i, path_meta in tqdm(enumerate(p.glob('*.txt'))):
        try:
            with open(str(path_meta), 'r') as f:
                meta = f.readlines()
                steps[i] = int(meta[-2].strip('\n').split('=')[1])
                distance[i] = float(meta[-3].strip('\n').split('=')[1])
                stat = meta[-4].strip('\n').split('=')[1]
                status[i] = np.equal(np.repeat(stat, 3), status_codes)
        except IndexError:
            pass
    num_no_traversable_start = status[:, 2].sum()
    num_good_samples = status.shape[0]-num_no_traversable_start-4
    results_status = (status.sum(0) / num_good_samples) * 100
    print(
        f'total num samples: {status.shape[0]}, {num_good_samples}, num no traversable start: {num_no_traversable_start}\n'
        f'No traversable start samples removed for result cal\n'
        f'{status_codes[0]}={results_status[0]:.3f}, mean_dist={distance[status[:, 0]].mean():3f}, mean_steps={steps[status[:, 0]].mean()}\n'
        f'{status_codes[1]}={results_status[1]:.3f}, mean_dist={distance[status[:, 1]].mean():3f}, mean_steps={steps[status[:, 1]].mean()}\n'
        # f'No traversable start samples for reference:\n'
        # f'{status_codes[2]}={results_status[2]:.3f}, mean_dist={distance[status[:, 2]].mean():3f}, mean_steps={steps[status[:, 2]].mean()}\n'
    )


if __name__ == "__main__":
    run()
