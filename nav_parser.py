import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='RoboHop')
    # dataset choices can only be hm3d or skokloster
    parser.add_argument('--dataset', '-ds', default='skokloster', choices=['hm3d','skokloster'], type=str, help='dataset to use')
    parser.add_argument('--seed', '-s', default=0, type=int, help='seed value')
    parser.add_argument('--path_rerun', '-pr', default=None, type=str, help='path to output dir that needs to be rerun')
    parser.add_argument('--max_steps', '-m', default=200, type=int, help='max number of steps')
    parser.add_argument('--use_depth', '-d', action='store_true', help='use depth for control')
    parser.add_argument('--use_depth4map', '-dm', action='store_true', help='use depth for mapping (TODO)')
    parser.add_argument('--resume_episode', '-re', default=None, type=str, help='dirpath containing episode.npy to resume an episode from its last state')
    parser.add_argument('--feed_states', '-fs', action='store_true', help='feed agent states (open-loop)')
    parser.add_argument('--mapIdx', '-i', default=0, type=int, help='map idx')


    parser.add_argument('--clip_csv', '-c', default="ceiling, floor, wall", type=str, help='remove nodes corresponding to these csv labels. Set it to "" to disable.')
    parser.add_argument('--greedy_propeller', '-gp', action='store_true', help='use greedy propeller: greedily localizes against a future image (with pros and cons)')
    parser.add_argument('--weight_string', '-ws', default=None, type=str, help='Set to "margin" to use this as weights to compute path lengths. The specified string must be an edge attribute in the topological graph')
    parser.add_argument('--fixed_lin_vel', '-v', default=None, type=float, help='Fixed linear velocity')
    parser.add_argument('--plan_da_nbrs', '-pn', action='store_true', help='compute path length as min over DA neighbors of the matched ref of a qry node')

    parser.add_argument('--controlModelPath', '-cmp', default=None, type=str, help='path to control model')

    if args is None:
        args = parser.parse_args()
    elif len(args) == 0:
        args = parser.parse_args([])
    else:
        args = parser.parse_args(args)
    return args

