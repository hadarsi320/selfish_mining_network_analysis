import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from main import mining_simulation
from utils import convert_args_dict


def args_to_outf(args, flags):
    return Path(
        'outputs/{tie-breaking}_{}_{pool-connectivity}_{pool-sizes}_{pool-powers}_{seed}.json'
        .format('selfish' if 'selfish-mining' in flags else 'honest', **args))


def outf_to_args(outf):
    # TODO implement
    pass


def rr_by_power_exp():
    default_args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 2, 'prints': 'parallel'}
    default_flags = []
    args_lists = []
    for tie_breaking in ['first', 'random']:
        for selfish_mining in [False, True]:
            for size in [0.3]:
                for connectivity in [0.5]:
                    for power in np.arange(0.05, 0.5, 0.05).round(2):
                        for seed in range(5):
                            args = default_args.copy()
                            flags = default_flags.copy()

                            if selfish_mining:
                                flags.append('selfish-mining')
                            args['tie-breaking'] = tie_breaking
                            args['pool-sizes'] = size
                            args['pool-connectivity'] = connectivity
                            args['pool-powers'] = power
                            args['seed'] = seed
                            args['outf'] = args_to_outf(args, flags)
                            if args['outf'].exists():
                                print('Skipping experiment as outf exists', args['outf'])
                                continue

                            cmd_args = convert_args_dict(args, flags)
                            args_lists.append(cmd_args)

    print('There are', len(args_lists), 'runs in total')
    start = datetime.now()
    with Pool() as p:
        for _ in tqdm(p.map(mining_simulation, args_lists)):
            pass

    print('The total runtime is ', datetime.now() - start)


if __name__ == '__main__':
    rr_by_power_exp()
