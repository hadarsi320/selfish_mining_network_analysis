import networkx as nx
import distinctipy
import numpy as np
from matplotlib import pyplot as plt


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


def plot_expectation():
    plt.plot([0, 0.5], [0, 0.5], label='Expected', ls='--')
