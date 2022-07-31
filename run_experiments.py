from multiprocessing import Pool
from pathlib import Path

import numpy as np

from main import mining_simulation
from utils import convert_args_dict


def rr_by_power_exp():
    default_args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 5, 'pool-connectivity': 0.5, 'prints': 'parallel'}
    default_flags = []
    args_lists = []
    for tie_breaking in ['first', 'random']:
        for selfish_mining in [False]:
            for banning in [False]:
                for pool_power in np.arange(0.05, 0.5, 0.05):
                    for seed in range(5):
                        args = default_args.copy()
                        flags = default_flags.copy()

                        if selfish_mining:
                            flags.append('selfish-mining')
                        if banning:
                            flags.append('banning')
                        args['tie-breaking'] = tie_breaking
                        args['pool-powers'] = pool_power
                        args['seed'] = seed
                        args['outf'] = Path('outputs/{tie-breaking}/{}/{pool-powers}/{seed}/out.json'.format(
                            'selfish' if selfish_mining else 'honest', **args))

                        # print('Running experiment number {} with outf {}'.format(i, args['outf']))
                        if args['outf'].exists():
                            print('Skipping experiment as outf exists', args['outf'])
                            continue

                        cmd_args = convert_args_dict(args, flags)
                        args_lists.append(cmd_args)
    with Pool(5) as p:
        p.map(mining_simulation, args_lists)


if __name__ == '__main__':
    rr_by_power_exp()
