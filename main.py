import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_relative_reward(power_list, rewards, selfish_pool=None):
    relative_rewards = rewards / rewards.sum()
    plt.plot([0, 1], [0, 1], label='Expected')
    if selfish_pool:
        plt.scatter(power_list[selfish_pool], relative_rewards[selfish_pool],
                    label='Selfish Pools', color='red')
        del power_list[selfish_pool]
        relative_rewards = np.delete(relative_rewards, selfish_pool)
    plt.scatter(power_list, relative_rewards, label='Honest Pools')
    plt.legend()
    plt.xlabel('Pool Power')
    plt.ylabel('Relative Reward')
    plt.show()


def generate_network_and_pools(N: int, n_k: list, p_k: list = None):
    """
    Generates a graph and distributes mining power through the graph
    :param N: The total size of the graph
    :param n_k: A list of the sizes of the pools in the graph
    :param p_k:
    :return: An nx graph object of the entire network, and a list of subnetworks
    """
    assert sum(n_k) <= N
    if p_k is not None:
        assert sum(p_k) <= 1
        assert (sum(n_k) <= N) == (sum(p_k) <= 1)
        assert len(n_k) == len(p_k)

    if sum(n_k) < N:
        n_k.append(N - sum(n_k))
        if p_k is not None:
            p_k.append(1 - sum(p_k))

    # G = nx.fast_gnp_random_graph(N, 0.1)
    # G = nx.newman_watts_strogatz_graph(N, 4, 2)
    G = nx.powerlaw_cluster_graph(N, 4, 0.1)

    nodes = random.sample(G.nodes, len(G))
    pools = []
    for N_i in n_k:
        pools.append(nodes[:N_i])
        del nodes[:N_i]

    powers = np.random.random(len(G))

    if p_k:
        for pool, p in zip(pools, p_k):
            powers[pool] = powers[pool] / powers[pool].sum() * p
    else:
        powers /= powers.sum()

    nx.set_node_attributes(G, dict(zip(G, powers)), name='power')
    G_pools = [G.subgraph(pool) for pool in pools]
    pool_powers = []
    for pool in G_pools:
        pool_powers.append(powers[pool.nodes].sum())
    return G, G_pools, powers, pool_powers


def mine(mining_power: np.array, pools: list, min_time: int):
    rewards = np.zeros(len(pools))
    pool_assignments = {node: i for i, pool in enumerate(pools) for node in pool}
    # for pool in pools:
    #     for node in pool:
    #         node['blockchain'] = []

    time = 0
    forked = False
    while time < min_time or forked:
        mining_times = np.random.exponential(mining_power ** -1)
        ordered_miners = np.argsort(mining_times)
        miner = ordered_miners[0]
        mining_pool = pool_assignments[miner]
        rewards[mining_pool] += 1
        time += min(mining_times)
    relative_rewards = rewards / rewards.sum()
    return relative_rewards


def main():
    N = 200
    message_time = 0.01
    random.seed(42)
    np.random.seed(42)

    G, pools, node_powers, pool_powers = generate_network_and_pools(N, [20, 50, 100])
    rel_rewards = mine(node_powers, pools, 10000)
    plot_relative_reward(pool_powers, rel_rewards)

    # layout = nx.spring_layout(G)
    # colors = ['red', 'green', 'teal']
    # for i, pool in enumerate(pools):
    #     power = sum(nx.get_node_attributes(pool, 'power').values())
    #     print('pool', i, 'has power:', power)
    #     nx.draw_networkx_nodes(pool, layout, node_color=colors[i])
    # nx.draw_networkx_edges(G, layout)
    # plt.show()


if __name__ == '__main__':
    main()
