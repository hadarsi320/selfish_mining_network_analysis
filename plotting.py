import json

from matplotlib import pyplot as plt

import utils
from utils import collect_results, POWER_RANGE, args_to_outf


def plot_expectation():
    plt.plot([0, 0.5], [0, 0.5], label='Expected', ls='--')


def plot_figure1():
    """
    Title: Comparison of Honest and Selfish Mining
    """
    results = {}
    args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 2, 'prints': 'parallel',
            'tie-breaking': 'first', 'pool-sizes': 0.5, 'pool-connectivity': 0.5}

    for selfish_mining in [False, True]:
        name = 'Selfish' if selfish_mining else 'Honest'
        results[name] = {}
        flags = []
        if selfish_mining:
            flags.append('selfish-mining')
        for power in POWER_RANGE:
            args['pool-powers'] = power
            results[name][power] = []
            for seed in range(5):
                args['seed'] = seed
                outf = args_to_outf(args, flags)
                run_result = json.load(outf.open())
                results[name][power].append(run_result['relative_rewards'][0])
    plot_expectation()
    utils.plot_dict(results)
    plt.xlabel('Mining Power')
    plt.ylabel('Relative Revenue')
    plt.savefig('plots/figure1.png')
    plt.show()


def plot_figure2():
    """
    Title: The affect of pool size and connectivity on threshold
    :return:
    """
    results = {}
    args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 2, 'prints': 'parallel',
            'tie-breaking': 'first'}
    flags = ['selfish-mining']

    for size in [0.1, 0.5, 0.9]:
        for connectivity in [0.1, 0.5, 0.9]:
            if size != 0.5 and connectivity != 0.5:
                continue
            if size < 0.5:
                name = 'Small Size'
            elif size > 0.5:
                name = 'Large Size'
            elif connectivity < 0.5:
                name = 'Low Connectivity'
            elif connectivity > 0.5:
                name = 'High Connectivity'
            else:
                name = 'Baseline'
            args['pool-sizes'] = size
            args['pool-connectivity'] = connectivity
            results[name] = collect_results(args, flags)

    plot_expectation()
    utils.plot_dict(results)
    plt.xlabel('Mining Power')
    plt.ylabel('Relative Revenue')
    plt.savefig('plots/figure2.png')
    plt.show()


def plot_figure3():
    """
    Title: The affect of the tie breaking method on threshold
    """
    results = {}
    args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 2, 'prints': 'parallel',
            'tie-breaking': None, 'pool-sizes': None, 'pool-connectivity': None}
    flags = ['selfish-mining']

    for tie_breaking in ['first', 'random']:
        args['tie-breaking'] = tie_breaking
        for size, connectivity in [(0.1, 0.1), (0.9, 0.9)]:
            name = tie_breaking.title() + ' Tie Breaking'
            if size == 0.1:
                name += ' Low $\gamma$'
            else:
                name += ' High $\gamma$'
            args['pool-sizes'] = size
            args['pool-connectivity'] = connectivity
            results[name] = collect_results(args, flags)

    plot_expectation()
    utils.plot_dict(results)
    plt.xlabel('Mining Power')
    plt.ylabel('Relative Revenue')
    plt.savefig('plots/figure3.png')
    plt.show()


if __name__ == '__main__':
    plot_figure1()
    plot_figure2()
    plot_figure3()
