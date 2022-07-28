import random
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Block:
    def __init__(self, creator, bid):
        self.creator = creator
        self.id = bid

    def __str__(self):
        return f'{self.id} ({self.creator})'

    def __repr__(self):
        return self.__str__()


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
    G = nx.powerlaw_cluster_graph(N, 2, 0.1)

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


def mine(G: nx.Graph, pools: List[nx.Graph], min_time: int, edge_time, tie_breaking='first'):
    def init_actions(t):
        if t not in actions:
            actions[t] = {'mine': [], 'receive': [], 'pending': []}

    def send_message(node, origin=None):
        # this function alerts other miners of the creation of a new block
        receive_time = t + edge_time
        init_actions(receive_time)
        neighbors = set(G.neighbors(node))
        if origin is not None:
            neighbors -= set([origin])
            if n2p[node] != n2p[origin]:
                neighbors = (neighbors | n2p[node].nodes) - set([node])

        for neighbor in neighbors:
            G.nodes[neighbor]['message'] = G.nodes[node]['blockchain'].copy()
            G.nodes[neighbor]['sender'] = node
            actions[receive_time]['receive'].append(neighbor)

    n2p = {node: pool for pool in pools for node in pool}
    for pool in pools:
        for node in pool:
            G.nodes[node]['blockchain'] = [Block(None, 0)]
            G.nodes[node]['seen'] = []

    last_block_id = 0
    actions = {0: {'pending': [node for node in G], 'receive': [], 'mine': []}}
    last_blocks = [0 for _ in G]
    n2mt = {}
    t = 0
    forked = False
    start = time.time()
    while t < min_time or forked:
        t = min(actions)
        frac = t / min_time
        if frac > 0:
            sys.stdout.write('\r[{}{}] {:.2f}/{} [{:.2f}%] {:.2f}s'
                             .format(int(frac * 50) * '#', int((1 - frac) * 50) * '-',
                                     t, min_time, frac * 100,
                                     time.time() - start))

        for node in actions[t]['mine']:
            last_block_id += 1
            new_block = Block(node, last_block_id)
            G.nodes[node]['blockchain'].append(new_block)
            G.nodes[node]['seen'].append(last_block_id)
            last_blocks[node] = last_block_id
            send_message(node)
            actions[t]['pending'].append(node)

        for node in actions[t]['receive']:
            # do 'message' and 'sender' need to be cleared?
            node_attr = G.nodes[node]
            new_blockchain = node_attr['message']
            node_blockchain = node_attr['blockchain']
            block_id = new_blockchain[-1].id
            if block_id not in node_attr['seen']:
                node_attr['seen'].append(block_id)
                if len(new_blockchain) > len(node_blockchain):
                    accept = True
                elif len(new_blockchain) == len(node_blockchain) and tie_breaking == 'random':
                    accept = random.random() > 0.5
                else:
                    accept = False

                if accept:
                    G.nodes[node]['blockchain'] = new_blockchain
                    send_message(node, G.nodes[node]['sender'])
                    actions[t]['pending'].append(node)
                    if node in n2mt:
                        actions[n2mt[node]]['mine'].remove(node)
                        n2mt.pop(node)
                    last_blocks[node] = block_id

        mining_power = np.array([G.nodes[node]['power'] for node in actions[t]['pending']])
        mining_times = np.random.exponential(mining_power ** -1)
        for node, compute_time in zip(actions[t]['pending'], mining_times):
            mine_time = t + compute_time
            init_actions(mine_time)
            actions[mine_time]['mine'].append(node)
            n2mt[node] = mine_time
        actions.pop(t)
        forked = any(item != last_blocks[0] for item in last_blocks)

    rewards = dict.fromkeys(pools, 0)
    blockchain = G.nodes[0]['blockchain']
    for block in blockchain[1:]:  # ignoring the 1st block
        rewards[n2p[block.creator]] += 1
    rewards = np.array(list(rewards.values()))
    relative_rewards = rewards / rewards.sum()
    return relative_rewards


def main():
    N = 100
    message_time = 0.01
    T = 1000
    random.seed(42)
    np.random.seed(42)

    G, pools, node_powers, pool_powers = generate_network_and_pools(N, [30, 20, 15])
    rel_rewards = mine(G, pools, T, message_time)
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
