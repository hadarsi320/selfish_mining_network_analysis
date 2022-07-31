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


def print_progress(t, min_time, start, forked, dynamic_progress=True):
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

        if dynamic_progress:
            sys.stdout.write('\r' + message)
        else:
            foobar = math.floor(frac * 100 / 5)
            if foobar > LAST:
                LAST = foobar
                print(message)
