import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from networkx import Graph

from plotting import draw_graph, plot_relative_reward
from utils import sample_sum_to, get_connectivity, print_progress


class Block:
    def __init__(self, creator, bid):
        self.creator = creator
        self.id = bid

    def __str__(self):
        return f'{self.id} ({self.creator})'

    def __repr__(self):
        return self.__str__()


def assert_pool_connected(G: Graph, pool: list):
    Gp = G.subgraph(pool)
    if not nx.is_connected(Gp):
        con_comp = iter(nx.connected_components(Gp))
        main_cc = next(con_comp)
        for cc in con_comp:
            u = random.sample(list(main_cc), 1)[0]
            v = random.sample(list(cc), 1)[0]
            G.add_edge(u, v)


def generate_network_and_pools(num_nodes: int, num_pools: int, pool_powers: list = None, pool_sizes: list = None,
                               pool_connectivity: float = 0, selfish_mining=False):
    """
    Generates a graph and distributes mining power through the graph
    :param num_nodes: The total size of the graph
    :param num_pools: The number of pools in the network, must be at least 1, as the last pool is not an actual pool
    and is just the rest of the nodes
    :param pool_powers: An optional list of the strength of the pools
    :param pool_sizes: A list of the sizes of the pools in the graph
    :param pool_connectivity: ??
    :param selfish_mining: Whether the first pool is doing selfish mining
    :return: An nx graph object of the entire network, and a list of sub graphs for each pool
    """
    if selfish_mining:
        assert num_pools > 1, 'Selfish mining only makes sense when pools exist'

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

    assert 0 <= pool_connectivity <= 1, 'Pool connectivity is a factor between 0 and 1'

    G = nx.powerlaw_cluster_graph(num_nodes, 2, 0.1)  # TODO set args

    nodes = random.sample(list(G.nodes), len(G))
    pools = []
    for i, pool_size in enumerate(pool_sizes):
        if i < num_pools - 1:
            pool_size = max(int(pool_size * num_nodes), 1)
            pools.append(nodes[:pool_size])
            del nodes[:pool_size]
        else:
            pools.append(nodes)
    pool_sizes = [len(pool) for pool in pools]

    assert sum(pool_sizes) == num_nodes
    assert all(pool_size != 0 for pool_size in pool_sizes)

    for i, (pool_power, pool_size) in enumerate(zip(pool_powers, pool_sizes)):
        logging.info(f'Pool {i + 1} has {pool_size} nodes and power {pool_power:.2f}')

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
                assert_pool_connected(G, pool)
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

    for node in G:
        if selfish_mining and node in G_pools[0]:
            G.nodes[node]['selfish'] = True
            G.nodes[node]['lead'] = 0
        else:
            G.nodes[node]['selfish'] = False

    return G, G_pools, powers, pool_powers, pool_sizes


def mine(G: nx.Graph, pools: List[nx.Graph], min_time: int, max_time: int, message_time, tie_breaking,
         prints, eps=1e-3):
    """
    :param G:
    :param pools: A list of the pools in the network, only used for reward aggregation
    :param min_time:
    :param max_time:
    :param message_time:
    :param tie_breaking:
    :param eps:
    :return:
    """
    last_block_id = 0
    n2p = {node: pool for pool in pools for node in pool}

    def init_actions(t):
        if t not in actions:
            actions[t] = {'mine': [], 'receive': [], 'pending': []}

    def pass_blockchain(node, message, only_in_pool):
        # this function alerts other miners of the creation of a new block
        receive_time = round(t + message_time, ndigits)
        init_actions(receive_time)
        neighbors = set(G.neighbors(node))
        if only_in_pool:
            neighbors = [neighbor for neighbor in neighbors if n2p[neighbor] == n2p[node]]
        block_id = message[-1].id

        for neighbor in neighbors:
            neighbor_attr = G.nodes[neighbor]
            if block_id not in neighbor_attr['seen']:
                if receive_time not in neighbor_attr['messages']:
                    neighbor_attr['messages'][receive_time] = [(message.copy(), node)]
                    actions[receive_time]['receive'].append(neighbor)
                elif all(message[-1].id != blockchain[-1].id
                         for blockchain, _ in neighbor_attr['messages'][receive_time]):
                    neighbor_attr['messages'][receive_time].append((message.copy(), node))

    def mine_block(node, block_id):
        attr = G.nodes[node]
        attr['blockchain'].append(Block(node, block_id))
        attr['seen'].append(block_id)
        last_blocks[node] = block_id
        if attr['selfish']:
            attr['lead'] += 1
        pass_blockchain(node, attr['blockchain'], attr['selfish'])
        actions_t['pending'].append(node)

    def receive_block():
        def accept(new_blockchain, block_id):
            # TODO change behavior if block is selfish
            attr = G.nodes[node]
            old_blockchain = attr['blockchain']
            attr['blockchain'] = new_blockchain
            if attr['selfish'] and n2p[new_blockchain[-1].creator] == n2p[node]:
                if len(old_blockchain) < len(new_blockchain):
                    assert G.nodes[sender]['lead'] >= attr['lead'] + 1
                    attr['lead'] += 1
                pass_blockchain(node, new_blockchain, True)
            else:
                pass_blockchain(node, new_blockchain, False)

            actions_t['pending'].append(node)
            last_blocks[node] = block_id
            if node in n2mt:
                actions[n2mt[node]]['mine'].remove(node)
                if all(len(l) == 0 for l in actions[n2mt[node]].values()):
                    actions.pop(n2mt[node])
                n2mt.pop(node)

        attr = G.nodes[node]
        messages = attr['messages'].pop(t)
        for new_blockchain, sender in messages:
            node_blockchain = attr['blockchain']
            block_id = new_blockchain[-1].id
            if block_id not in attr['seen']:
                attr['seen'].append(block_id)
                if len(new_blockchain) > len(node_blockchain):
                    accept(new_blockchain, block_id)

                elif len(new_blockchain) == len(node_blockchain):
                    if attr['selfish']:
                        if n2p[new_blockchain[-1].creator] == n2p[node]:
                            accept(new_blockchain, block_id)
                        elif attr['lead'] == 1:
                            attr['lead'] = 0
                            pass_blockchain(node, node_blockchain, False)
                            pass_blockchain(node, new_blockchain, True)
                    elif tie_breaking == 'random' and random.random() > 0.5:
                        accept(new_blockchain, block_id)

                elif attr['selfish']:
                    if attr['lead'] >= 1:
                        if attr['lead'] > 2:
                            attr['lead'] -= 1
                            revealed_blockchain = node_blockchain[:-attr['lead']]
                        else:
                            attr['lead'] = 0
                            revealed_blockchain = node_blockchain
                        pass_blockchain(node, revealed_blockchain, False)
                        pass_blockchain(node, new_blockchain, True)

    def sample_mining_times():
        mining_power = np.array([G.nodes[node]['power'] for node in actions_t['pending']])
        mining_times = np.random.exponential(mining_power ** -1)
        for node, compute_time in zip(actions_t['pending'], mining_times):
            mine_time = round(t + compute_time, ndigits)
            init_actions(mine_time)
            actions[mine_time]['mine'].append(node)
            n2mt[node] = mine_time

    ndigits = -int(math.log10(eps))
    for node in G:
        attr = G.nodes[node]
        attr['blockchain'] = [Block(None, 0)]
        attr['seen'] = []
        attr['messages'] = {}

    actions = {0: {'pending': [node for node in G], 'receive': [], 'mine': []}}
    history = {}
    last_blocks = [0 for _ in G]
    n2mt = {}  # node to mine time

    times = iter(np.arange(0, min_time * 2, eps).round(ndigits))
    t = 0
    forked = False
    forked_time = 0
    start = time.time()
    while t < min_time or (t < max_time and forked):
        t = next(times)
        if t in actions:
            actions_t = actions.pop(t)
            history[t] = actions_t.copy()
            print_progress(t, min_time, start, False, prints=prints)
            for node in actions_t['mine']:
                last_block_id += 1
                mine_block(node, last_block_id)
            for node in actions_t['receive']:
                receive_block()
            sample_mining_times()
            history[t].update(actions_t)

            forked = any(item != last_blocks[0] for item in last_blocks)
            if forked:
                forked_time += eps
    print()
    finish = t
    logging.info('The blockchain was forked {:.2f}% of the time'.format(forked_time / finish * 100))

    rewards = dict.fromkeys(pools, 0)
    blockchain = G.nodes[0]['blockchain']
    for block in blockchain[1:]:  # ignoring the 1st block
        rewards[n2p[block.creator]] += 1
    rewards = np.array(list(rewards.values()))
    relative_rewards = rewards / rewards.sum()
    return relative_rewards, forked_time / finish


def parse_args(input: list):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-nodes', type=int, default=1000)
    parser.add_argument('-T', '--turns', type=int, default=1000)
    parser.add_argument('-p', '--num-pools', type=int)
    parser.add_argument('--pool-powers', type=float, nargs='*')
    parser.add_argument('--pool-sizes', type=float, nargs='*')
    parser.add_argument('--pool-connectivity', type=float, default=0)

    parser.add_argument('--message-time', type=float, default=0.01)
    parser.add_argument('--tie-breaking', type=str, choices=['first', 'random'], default='first')
    parser.add_argument('--selfish-mining', action='store_true')  # TODO implement
    parser.add_argument('--banning', action='store_true')  # TODO implement

    parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('--outf', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--plot-graph', action='store_true')
    parser.add_argument('--prints', choices=['dynamic', 'update', 'parallel'])

    args = parser.parse_args(args=input)

    if args.debug:
        args.outf = Path('outputs/debug.json')
        if args.outf.exists():
            args.outf.unlink()
    elif args.outf is None:
        args.outf = Path(f'outputs/{args.num_nodes}_{args.num_pools}_{args.seed}.json')
    args.outf.parent.mkdir(parents=True, exist_ok=True)

    return args


def mining_simulation(input=None):
    args = parse_args(input)

    random.seed(args.seed)
    np.random.seed(args.seed)

    G, pools, node_powers, pool_powers, pool_sizes = \
        generate_network_and_pools(args.num_nodes, args.num_pools, args.pool_powers, args.pool_sizes,
                                   args.pool_connectivity, args.selfish_mining)
    if args.debug:
        draw_graph(G, pools)

    rel_rewards, forked_time = mine(G, pools, args.turns, args.turns * 1.1, args.message_time, args.tie_breaking,
                                    prints=args.prints)

    output = {key: str(value) if isinstance(value, Path) else value for key, value in args.__dict__.items()}
    output.update({'pool_powers': pool_powers, 'pool_sizes': pool_sizes,
                   'relative_rewards': rel_rewards.tolist(), 'forked_time': forked_time})
    json.dump(output, args.outf.open('w'), indent=4)

    if args.prints == 'parallel':
        print('Finished run, saved at', args.outf)

    if args.plot_graph or args.debug:
        plot_relative_reward(pool_powers, rel_rewards, args.selfish_mining)


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    mining_simulation()
