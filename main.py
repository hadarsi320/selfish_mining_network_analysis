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
from utils import sample_sum_to, print_progress


class Block:
    def __init__(self, creator, id, t):
        self.creator = creator
        self.id = str(id)
        self.time = t

    def __str__(self):
        return f'Block({self.id}, {self.creator}, {self.time})'

    def __repr__(self):
        return self.__str__()


class Message:
    def __init__(self, blockchain, sender, flag=''):
        self._blockchain = blockchain.copy()
        self.sender = sender
        self.flag = flag

    def __str__(self):
        repr = f'{self._blockchain} [{self.sender}]'
        if self.flag:
            repr += f' ({self.flag})'
        return repr

    def __repr__(self):
        return self.__str__()

    @property
    def blockchain(self):
        return self._blockchain.copy()

    @property
    def id(self):
        return self._blockchain[-1].id + self.flag


def assert_pool_connected(G: Graph, pool: list):
    Gp = G.subgraph(pool)
    if not nx.is_connected(Gp):
        con_comp = iter(nx.connected_components(Gp))
        main_cc = next(con_comp)
        for cc in con_comp:
            u = random.sample(list(main_cc), 1)[0]
            v = random.sample(list(cc), 1)[0]
            G.add_edge(u, v)


def generate_network_and_pools(num_nodes: int, num_pools: int, graph_args, pool_powers: list = None,
                               pool_sizes: list = None, pool_connectivity: float = 0, selfish_mining=False):
    """
    Generates a graph and distributes mining power through the graph
    :param num_nodes: The total size of the graph
    :param num_pools: The number of pools in the network, must be at least 1, as the last pool is not an actual pool
    and is just the rest of the nodes
    :param pool_powers: An optional list of the strength of the pools
    :param pool_sizes: A list of the sizes of the pools in the graph
    :param pool_connectivity: The connectivity of the nodes in the pool to the coordinator
    :param selfish_mining: Whether the first pool is performing selfish mining
    :param graph_args: Arguments for graph creation
    :return: An nx graph object of the entire network, and a list of sub graphs for each pool
    """
    assert num_pools >= 1
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

    assert 0 < pool_connectivity <= 1, 'The coordinator cannot be disconnected from the pool'

    G: Graph = nx.powerlaw_cluster_graph(num_nodes, int(graph_args[0]), float(graph_args[1]))
    assert nx.is_connected(G), 'The bitcoin network must be connected'

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

    # set powers
    powers = np.random.random(num_nodes)
    for pool, pool_power in zip(pools, pool_powers):
        powers[pool] = powers[pool] / powers[pool].sum() * pool_power

    nx.set_node_attributes(G, dict(zip(G, powers)), name='power')

    # if the first pool performs selfish mining, it must have a coordinator
    if selfish_mining:
        assert num_pools > 1, 'If there are no pools, there can\'t be selfish mining'
        coordinator = f'coordinator'
        G.add_node(coordinator)
        num_edges = int(pool_connectivity * len(pools[0]))
        assert num_edges > 0
        connected_nodes = random.sample(pools[0], num_edges)
        pools[0].append(coordinator)
        G.add_edges_from([(coordinator, node) for node in connected_nodes])
        logging.info('The selfish pool has {:,} miners and {:,} connections to its coordinator'
                     .format(len(pools[0]) - 1, num_edges))
        G.nodes['coordinator']['lead'] = 0

    G_pools = [nx.subgraph(G, pool) for pool in pools]
    for node in G:
        G.nodes[node]['selfish'] = selfish_mining and node in G_pools[0]

    return G, G_pools, powers, pool_powers, pool_sizes


def get_ids(cur_blockchain, new_blockchain):
    return [block.id for block in new_blockchain[len(cur_blockchain) - 1:]]


def compute_relative_rewards(G, pools, n2p):
    rewards = dict.fromkeys(pools, 0)
    blockchain = G.nodes[0]['blockchain']
    for block in blockchain[1:]:  # ignoring the 1st block
        rewards[n2p[block.creator]] += 1
    rewards = np.array(list(rewards.values()))
    relative_rewards = rewards / rewards.sum()
    return relative_rewards


def update_public_branch(node_attributes, new_blockchain):
    assert len(new_blockchain) - len(node_attributes['public_branch']) == 1, \
        'The actual public branch is of length {} while this node thought it was of length {}'.format(
            len(new_blockchain), len(node_attributes['public_branch']) + 1)
    node_attributes['public_branch'] = new_blockchain.copy()


def get_miners(G):
    return [node for node in G if not str(node).startswith('coordinator')]


def is_coordinator(node):
    return node == 'coordinator'


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
    :param prints:
    :return:
    """
    last_block_id = 0
    n2p = {node: pool for pool in pools for node in pool}
    ndigits = -int(math.log10(eps))
    for node in G:
        attr = G.nodes[node]
        attr['seen'] = []
        attr['messages'] = {}
        attr['blockchain'] = [Block(None, 0, 0)]  # this is the genesis block
        if not attr['selfish'] or is_coordinator(node):
            attr['last_block'] = 0
            if is_coordinator(node):
                attr['public_branch'] = [Block(None, 0, 0)]

    actions = {0: {'pending': get_miners(G), 'receive': [], 'mine': []}}
    history = {}
    n2mt = {}  # node to mine time

    def init_actions(t):
        if t not in actions:
            actions[t] = {'mine': [], 'receive': [], 'pending': set()}

    def same_pool(u, v):
        return n2p[u] == n2p[v]

    def cancel_mine(node):
        actions[n2mt[node]]['mine'].remove(node)
        if all(len(l) == 0 for l in actions[n2mt[node]].values()):
            actions.pop(n2mt[node])
        n2mt.pop(node)

    def pass_blockchain(sender, new_blockchain, only_pool=False, flag=''):
        # this function alerts other miners of the creation of a new block
        receive_time = round(t + message_time, ndigits)
        init_actions(receive_time)
        neighbors = set(G.neighbors(sender))
        if only_pool:
            neighbors = [neighbor for neighbor in neighbors if same_pool(neighbor, sender)]
        block_id = new_blockchain[-1].id

        for neighbor in neighbors:
            neighbor_attr = G.nodes[neighbor]
            nflag = flag if neighbor_attr['selfish'] and not is_coordinator(neighbor) else ''
            message = Message(new_blockchain, sender, nflag)
            if message.id not in neighbor_attr['seen']:
                if receive_time not in neighbor_attr['messages']:
                    neighbor_attr['messages'][receive_time] = [message]
                    actions[receive_time]['receive'].append(neighbor)
                elif all(block_id != m.blockchain[-1].id
                         for m in neighbor_attr['messages'][receive_time]):
                    neighbor_attr['messages'][receive_time].append(message)
                if same_pool(neighbor, new_blockchain[-1].creator) and neighbor in n2mt and \
                        n2mt[neighbor] == receive_time:
                    cancel_mine(neighbor)

    def mine_block(node, block_id):
        new_block = Block(node, block_id, t)
        attr = G.nodes[node]
        attr['blockchain'].append(new_block)
        attr['seen'].append(block_id)
        if not attr['selfish']:
            attr['last_block'] = block_id
        pass_blockchain(node, attr['blockchain'], only_pool=attr['selfish'])
        actions_t['pending'].add(node)
        n2mt.pop(node)

    def receive_messages():
        def accept():
            """
            When this function runs, the current node accepts a received blockchain as the head of the blockchain
             and mines on it
            """
            attr['blockchain'] = new_blockchain
            attr['last_block'] = new_blockchain[-1].id
            if not is_coordinator(node):
                actions_t['pending'].add(node)
            if node in n2mt:
                cancel_mine(node)

        attr = G.nodes[node]
        messages = attr['messages'].pop(t)
        for message in messages:
            new_blockchain = message.blockchain
            if attr['selfish'] and not is_coordinator(node) and message.id not in attr['seen']:
                attr['seen'].append(message.id)
                attr['blockchain'] = message.blockchain
                pass_blockchain(node, new_blockchain, only_pool=message.flag != 'leak', flag=message.flag)

            else:
                cur_blockchain = attr['blockchain']
                creator = new_blockchain[-1].creator
                created_by_pool = same_pool(creator, node)
                if message.id not in attr['seen']:
                    # attr['seen'].extend(get_ids(cur_blockchain, new_blockchain))
                    attr['seen'].append(message.id)

                    if len(new_blockchain) > len(cur_blockchain):
                        if is_coordinator(node):
                            if created_by_pool:
                                assert len(new_blockchain) == len(cur_blockchain) + 1
                                attr['lead'] += 1
                            else:
                                assert attr['lead'] == 0
                                update_public_branch(attr, new_blockchain)
                                pass_blockchain(node, new_blockchain, flag='leak')
                        else:
                            pass_blockchain(node, new_blockchain)
                        accept()

                    elif len(new_blockchain) == len(cur_blockchain):
                        if is_coordinator(node):
                            if created_by_pool:
                                accept()
                            elif attr['lead'] == 1:
                                attr['lead'] -= 1
                                update_public_branch(attr, new_blockchain)
                                pass_blockchain(node, cur_blockchain, flag='leak')
                        elif tie_breaking == 'random' and random.random() > 0.5:
                            accept()
                            pass_blockchain(node, new_blockchain, False)

                    elif is_coordinator(node):
                        if not created_by_pool and len(new_blockchain) > len(attr['public_branch']):
                            update_public_branch(attr, new_blockchain)
                            assert attr['lead'] >= 2
                            if attr['lead'] == 2:
                                attr['lead'] = 0
                                revealed_blockchain = cur_blockchain
                                update_public_branch(attr, revealed_blockchain)
                            else:
                                attr['lead'] -= 1
                                revealed_blockchain = cur_blockchain[:-attr['lead']]
                            pass_blockchain(node, revealed_blockchain, flag='leak')

    def sample_mining_times():
        mining_power = np.array([G.nodes[node]['power'] for node in actions_t['pending']])
        mining_times = np.random.exponential(mining_power ** -1)
        for node, compute_time in zip(actions_t['pending'], mining_times):
            mine_time = round(t + compute_time, ndigits)
            init_actions(mine_time)
            actions[mine_time]['mine'].append(node)
            n2mt[node] = mine_time

    times = iter(np.arange(0, min_time * 2, eps).round(ndigits))
    t = 0
    forked = False
    forked_time = 0
    start = time.time()
    while t < min_time or (t < max_time and forked):
        t = next(times)
        if t in actions:
            actions_t = actions.pop(t)
            history[t] = actions_t
            print_progress(t, min_time, start, False, prints=prints)
            for node in actions_t['mine']:
                last_block_id += 1
                mine_block(node, str(last_block_id))
            for node in actions_t['receive']:
                receive_messages()
            sample_mining_times()

            last_block = next(iter(nx.get_node_attributes(G, 'last_block').values()))
            forked = any(block_id != last_block for block_id in
                         nx.get_node_attributes(G, 'last_block').values())
            if forked:
                forked_time += eps
    print()
    finish = t
    logging.info('The blockchain was forked {:.2f}% of the time'.format(forked_time / finish * 100))

    rr = compute_relative_rewards(G, pools, n2p)
    return rr, forked_time / finish


def parse_args(input: list):
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-nodes', type=int, default=1000)
    parser.add_argument('--graph-args', nargs=2, default=[2, 0.5])
    parser.add_argument('-T', '--turns', type=int, default=1000)
    parser.add_argument('-p', '--num-pools', type=int)
    parser.add_argument('--pool-powers', type=float, nargs='*')
    parser.add_argument('--pool-sizes', type=float, nargs='*')
    parser.add_argument('--pool-connectivity', type=float, default=0)

    parser.add_argument('--message-time', type=float, default=0.01)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--tie-breaking', type=str, choices=['first', 'random'], default='first')
    parser.add_argument('--selfish-mining', action='store_true')
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
        generate_network_and_pools(args.num_nodes, args.num_pools, args.graph_args, args.pool_powers, args.pool_sizes,
                                   args.pool_connectivity, args.selfish_mining)
    if args.debug:
        draw_graph(G, pools)

    rel_rewards, forked_time = mine(G, pools, args.turns, args.turns * 1.1, args.message_time, args.tie_breaking,
                                    prints=args.prints, eps=args.eps)

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
