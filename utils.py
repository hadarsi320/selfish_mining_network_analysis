import json
import math
import sys
import time
from pathlib import Path

import distinctipy
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


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


def draw_graph(G, pools):
    layout = nx.spring_layout(G)
    colors = distinctipy.get_colors(len(pools))
    for i, (pool, color) in enumerate(zip(pools, colors)):
        nx.draw_networkx_nodes(pool, layout, node_color=(color,), label=f'Pool {i + 1}', node_size=100)
    nx.draw_networkx_edges(G, layout)
    plt.legend(loc='best')
    plt.show()


def plot_relative_reward(power_list, rewards, selfish_mining=False):
    relative_rewards = rewards / rewards.sum()
    plt.plot([0, 1], [0, 1], label='Expected')
    if selfish_mining:
        plt.scatter(power_list[1:], relative_rewards[1:], label='Honest Pools')
        plt.scatter(power_list[0], relative_rewards[0], label='Selfish Pool', color='red')
    else:
        plt.scatter(power_list, relative_rewards, label='Pools')
    plt.legend()
    plt.xlabel('Pool Power')
    plt.ylabel('Relative Reward')
    plt.show()


def plot_dict(results):
    for key in results:
        x = results[key].keys()
        y = [np.mean(l) for l in results[key].values()]
        plt.plot(x, y, label=key)
    plt.legend()


def collect_results(args, flags):
    results = {}
    for power in POWER_RANGE:
        args['pool-powers'] = power
        results[power] = []
        for seed in range(5):
            args['seed'] = seed
            outf = args_to_outf(args, flags)
            run_result = json.load(outf.open())
            results[power].append(run_result['relative_rewards'][0])
    return results


POWER_RANGE = np.arange(0.05, 0.5, 0.05).round(2)


def args_to_outf(args, flags):
    return Path(
        'outputs/{tie-breaking}_{}_{pool-connectivity}_{pool-sizes}_{pool-powers}_{seed}.json'
        .format('selfish' if 'selfish-mining' in flags else 'honest', **args))
