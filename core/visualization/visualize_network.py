#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt

from core.configuration.consts import (
    LAYER_INTERVAL, NODE_INTERVAL, LAYER_TYPE_NAME2COLOR, VISUAL_WEIGHT_CONST,
    ar_type2sign
)


def visualize_network(network_layers, start_index=0, end_index=-1, title="figure 1", next_layer_part2union=None,
              out_image_path=None, debug=False) -> None:
    nn = nx.Graph()
    # adding nodes
    for i,layer in enumerate(network_layers[start_index:]):
        color = LAYER_TYPE_NAME2COLOR[layer.type_name]
        for j, node in enumerate(layer.nodes):
            nn.add_node(node.name,
                        pos=(i*LAYER_INTERVAL, j*NODE_INTERVAL),
                        label=ar_type2sign.get(node.ar_type, node.name),
                        color=node.name, # color,
                        style="filled",
                        fillcolor=color)
    # adding out_edges (no need to iterate over output layer)
    for i, layer in enumerate(network_layers[:end_index]):
        for j, node in enumerate(layer.nodes):
            for edge in node.out_edges:
                if next_layer_part2union is None or \
                        edge.dest not in next_layer_part2union:
                    dest = edge.dest
                else:
                    dest = next_layer_part2union[edge.dest]
                visual_weight = round(edge.weight, VISUAL_WEIGHT_CONST)
                nn.add_edge(edge.src, dest, weight=visual_weight)

    pos = nx.get_node_attributes(nn, 'pos')
    node_labels = nx.get_node_attributes(nn, 'label')
    node_colors = nx.get_node_attributes(nn, 'fillcolor').values()
#        node_colors = sorted(node_colors.values(), key=sorting_color_key)

    weights = nx.get_edge_attributes(nn,'weight')

    if debug:
        print("visualize() debug=True")
        import IPython
        IPython.embed()

    #nx.draw_networkx(nn,pos,edge_labels=labels, label_pos=0.8)
    plt.figure(title)
    # to set title inside the figure
    #nx.draw_networkx(nn,pos, title=title)
    nx.draw(nn,pos, title=title)
    nx.draw_networkx_nodes(nn,pos, node_color=node_colors)
    nx.draw_networkx_labels(nn,pos,labels=node_labels,node_color=node_colors)
    nx.draw_networkx_edge_labels(nn,pos,edge_labels=weights, label_pos=0.75)
    if out_image_path is not None:
        plt.savefig(out_image_path)
    else:
        plt.show()
