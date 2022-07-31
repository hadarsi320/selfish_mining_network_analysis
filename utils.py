import math
import sys
import time

import numpy as np


def sample_sum_to(size, sum):
    sizes = np.random.random(size)
    sizes = sizes / sizes.sum() * sum
    return sizes


def get_connectivity(graph):
    max_edges = math.comb(len(graph), 2)
    num_edges = len(graph.edges)
    connectivity = num_edges / max_edges
    return connectivity


LAST = 0


def print_progress(t, min_time, start, forked, prints=True):
    global LAST
    frac = t / min_time
    if frac > 0:
        message = '[{}{}] {:.2f}/{} [{:.2f}%] {}{:.2f}s'.format(
            min(round(frac * 50), 50) * '#',
            max(round((1 - frac) * 50), 0) * '-',
            t, min_time, frac * 100,
            'Forked ' if forked else '',
            time.time() - start,
        )

        if prints == 'dynamic':
            sys.stdout.write('\r' + message)
        elif prints == 'update':
            foobar = math.floor(frac * 100 / 5)
            if foobar > LAST:
                LAST = foobar
                print(message)


def convert_args_dict(args_dict: dict, flags=()):
    """This is a copy of another function in the project so no installed packages are required"""
    args = []
    for key, value in args_dict.items():
        key = '-'.join(key.split('_'))
        if ' ' in key:
            raise ValueError(f'Key "{key}" contains a space, which is not allowed')
        args.append(f'-{key}' if len(key) == 1 else f'--{key}')
        if isinstance(value, list):
            args.extend(value)
        else:
            args.append(value)
    for flag in flags:
        args.append(f'-{flag}' if len(flag) == 1 else f'--{flag}')
    return list(map(str, args))
