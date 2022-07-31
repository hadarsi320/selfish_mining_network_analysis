import networkx as nx
import distinctipy
from matplotlib import pyplot as plt


def draw_graph(G, pools):
    layout = nx.spring_layout(G)
    colors = distinctipy.get_colors(len(pools))
    for i, (pool, color) in enumerate(zip(pools, colors)):
        nx.draw_networkx_nodes(pool, layout, node_color=(color, ), label=f'Pool {i + 1}', node_size=100)
    nx.draw_networkx_edges(G, layout)
    plt.legend(loc='best')
    plt.show()


def plot_relative_reward(power_list, rewards, selfish_pool=False):
    relative_rewards = rewards / rewards.sum()
    plt.plot([0, 1], [0, 1], label='Expected')
    if selfish_pool:
        plt.scatter(power_list[0], relative_rewards[0],
                    label='Selfish Pools', color='red')
        plt.scatter(power_list[1:], relative_rewards[1:], label='Honest Pools' if selfish_pool else 'Pools')
    else:
        plt.scatter(power_list, relative_rewards, label='Pools')
    plt.legend()
    plt.xlabel('Pool Power')
    plt.ylabel('Relative Reward')
    plt.show()
