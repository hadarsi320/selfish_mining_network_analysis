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
