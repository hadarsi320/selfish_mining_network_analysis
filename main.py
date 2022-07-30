import argparse
import logging
import math
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


def sample_sum_to(size, sum):
    sizes = np.random.random(size)
    sizes = sizes / sizes.sum() * sum
    return sizes


def get_connectivity(graph):
    max_edges = math.comb(len(graph), 2)
    num_edges = len(graph.edges)
    connectivity = num_edges / max_edges
    return connectivity


def generate_network_and_pools(num_nodes: int, num_pools: int, pool_powers: list = None, pool_sizes: list = None,
                               pool_connectivity: float = 0):
    """
    Generates a graph and distributes mining power through the graph
    :param num_nodes: The total size of the graph
    :param num_pools: The number of pools in the network, must be at least 1, as the last pool is not an actual pool
    and is just the rest of the nodes
    :param pool_powers: An optional list of the strength of the pools
    :param pool_sizes: A list of the sizes of the pools in the graph
    :param pool_connectivity: ??
    :return: An nx graph object of the entire network, and a list of subgraphs for each pool
    """
    if pool_powers is None:
        pool_powers = []
    if pool_sizes is None:
        pool_sizes = []

    for _list in [pool_powers, pool_sizes]:
        assert len(_list) <= num_pools, 'The list cannot have more objects than the number of pools'
        if len(_list) < num_pools:
            assert sum(_list) < 1
            _list.extend(sample_sum_to(num_pools - len(_list), 1 - sum(_list)))
        elif len(_list) == num_pools:
            assert sum(_list) == 1
        else:
            raise ValueError('The list cannot have more objects than there are pools')

    assert 0 <= pool_connectivity <= 1

    G = nx.powerlaw_cluster_graph(num_nodes, 2, 0.1)  # TODO set args

    nodes = random.sample(list(G.nodes), len(G))
    pools = []
    for i, pool_size in enumerate(pool_sizes):
        if i < num_pools - 1:
            pool_size = int(pool_size * num_nodes)
            pools.append(nodes[:pool_size])
            del nodes[:pool_size]
        else:
            pools.append(nodes)

    powers = np.random.random(num_nodes)
    for pool, pool_power in zip(pools, pool_powers):
        powers[pool] = powers[pool] / powers[pool].sum() * pool_power

    nx.set_node_attributes(G, dict(zip(G, powers)), name='power')
    G_pools = [nx.subgraph(G, pool) for pool in pools]

    # the last pool is just the rest of the nodes and not a single entity, and therefore we don't require them to
    # have any
    for i, pool in enumerate(G_pools):
        if len(pool) > 1:
            max_edges = math.comb(len(pool), 2)
            if i < num_pools - 1:
                num_edges = len(pool.edges)
                if get_connectivity(pool) < pool_connectivity:
                    num_missing = int(max_edges * pool_connectivity) - num_edges
                    total_edges = nx.complete_graph(pool.nodes).edges
                    edges_to_add = random.sample(list(total_edges - pool.edges), num_missing)
                    G.add_edges_from(edges_to_add)
            logging.info('Pool {} has connectivity {:.3f}{}'.
                         format(i + 1, get_connectivity(pool),
                                ', it is the rest pseudo pool' * (i == num_pools - 1)))
        else:
            logging.info(f'Pool {i + 1} has a single node')
    logging.info('The total network connectivity is {:.3f}'.format(get_connectivity(G)))

    return G, G_pools, powers, pool_powers


LAST = 0


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


def mine(G: nx.Graph, pools: List[nx.Graph], min_time: int, max_time: int, message_time, tie_breaking,
         dynamic_progress=True, eps=1e-3):
    """

    :param G:
    :param pools: A list of the pools in the network, only used for reward aggregation
    :param min_time:
    :param max_time:
    :param message_time:
    :param tie_breaking:
    :param dynamic_progress:
    :param eps:
    :return:
    """

    def init_actions(t):
        if t not in actions:
            actions[t] = {'mine': [], 'receive': [], 'pending': []}

    def send_message(node, origin=None):
        # this function alerts other miners of the creation of a new block
        receive_time = t + message_time
        init_actions(receive_time)
        neighbors = set(G.neighbors(node))
        message = G.nodes[node]['blockchain'].copy()
        block_id = message[-1].id
        if origin is not None:
            neighbors.remove(origin)

        for neighbor in neighbors:
            neighbor_attr = G.nodes[neighbor]
            if block_id not in neighbor_attr['seen']:
                neighbor_attr['message'] = message
                neighbor_attr['sender'] = node
                actions[receive_time]['receive'].append(neighbor)

    ndigits = -int(math.log10(eps))
    for node in G:
        G.nodes[node]['blockchain'] = [Block(None, 0)]
        G.nodes[node]['seen'] = []

    last_block_id = 0
    actions = {0: {'pending': [node for node in G], 'receive': [], 'mine': []}}
    last_blocks = [0 for _ in G]
    n2mt = {}

    times = iter(np.arange(0, min_time * 2, eps).round(ndigits))
    t = 0
    forked = False
    forked_time = 0
    start = time.time()
    while t < min_time or (t < max_time and forked):
        t = next(times)
        if t in actions:
            actions_t = actions.pop(t)
            print_progress(t, min_time, start, False, dynamic_progress=dynamic_progress)
            for node in actions_t['mine']:
                last_block_id += 1
                new_block = Block(node, last_block_id)
                G.nodes[node]['blockchain'].append(new_block)
                G.nodes[node]['seen'].append(last_block_id)
                last_blocks[node] = last_block_id
                send_message(node)
                actions_t['pending'].append(node)

            for node in actions_t['receive']:
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
                        actions_t['pending'].append(node)
                        if node in n2mt:
                            actions[n2mt[node]]['mine'].remove(node)
                            n2mt.pop(node)
                        last_blocks[node] = block_id

            mining_power = np.array([G.nodes[node]['power'] for node in actions_t['pending']])
            mining_times = np.random.exponential(mining_power ** -1).round(ndigits)
            for node, compute_time in zip(actions_t['pending'], mining_times):
                mine_time = t + compute_time
                init_actions(mine_time)
                actions[mine_time]['mine'].append(node)
                n2mt[node] = mine_time
            forked = any(item != last_blocks[0] for item in last_blocks)
            if forked:
                forked_time += eps
    finish = t
    logging.info('{:.2f}% of the time the blockchain was forked'.format(forked_time / finish * 100))

    rewards = dict.fromkeys(pools, 0)
    blockchain = G.nodes[0]['blockchain']
    n2p = {node: pool for pool in pools for node in pool}
    for block in blockchain[1:]:  # ignoring the 1st block
        rewards[n2p[block.creator]] += 1
    rewards = np.array(list(rewards.values()))
    relative_rewards = rewards / rewards.sum()
    return relative_rewards, forked_time / finish


def parse_args(parser_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-nodes', type=int, default=1000)
    parser.add_argument('-T', '--turns', type=int, default=1000)
    parser.add_argument('--num-pools', type=int)
    parser.add_argument('--pool-powers', type=float, nargs='*')
    parser.add_argument('--pool-sizes', type=float, nargs='*')
    parser.add_argument('--pool-connectivity', type=float, default=0)

    parser.add_argument('--message-time', type=float, default=0.01)
    parser.add_argument('--tie-breaking', type=str, choices=['first', 'random'])
    parser.add_argument('--selfish-mining', action='store_true')  # TODO implement
    parser.add_argument('--banning', action='store_true')  # TODO implement

    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dynamic-progress', action='store_true')

    args = parser.parse_args(args=parser_args)

    return args


def main():
    args = parse_args(None)
    logging.getLogger().setLevel('INFO')

    random.seed(args.seed)
    np.random.seed(args.seed)

    G, pools, node_powers, pool_powers = generate_network_and_pools(args.num_nodes, args.num_pools, None,
                                                                    args.pool_powers,
                                                                    args.pool_connectivity)
    rel_rewards, forked_time = mine(G, pools, args.turns, args.turns * 1.1,
                                    args.message_time, args.tie_breaking,
                                    dynamic_progress=args.dynamic_progress)
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
