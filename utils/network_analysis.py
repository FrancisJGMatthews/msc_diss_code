import datetime
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from networkx.algorithms import bipartite


# ----------------------- BASIC NETWORK FUNCTIONS ----------------------- #


def getGiantComponent(G):
    """Return the giant connected component of a network object G"""
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    return G


# ----------------------- NETWORK PLOTTING FUNCTIONS ----------------------- #

def plotDegreeDistribution(nodes, label='nodes', title='title', color='red', figsize=(12,5),save=False):
    """Calculate and plot degree distribution for a node set, return degree distribution table. Plot both linear-linear scale and log-log scale

    Args:
        - nodes (dict) : dictionary object containing {node : degree} information for all nodes in set
        - label (str) : legend label for node set
        - color (str) : color of plot markers
        - save (bool) : flag to save figure (default False)
    Returns:
        - degree_distribution_table (pandas.core.frame.DataFrame) : python (sorted) dataframe with degree / node count columns for node set
    """

    # Calculate degree distribution
    degree_sequence = list(nodes.values())
    histogram_count = Counter(degree_sequence)
    degree_distribution_table = pd.DataFrame.from_records([(i, histogram_count[i]) for i in histogram_count], columns = ['Degree', 'Node_count'])
    total_nodes = degree_distribution_table['Node_count'].sum()
    degree_distribution_table['pk'] = degree_distribution_table['Node_count'] / total_nodes

    # Plot degree distribution
    with plt.style.context("ggplot"):
        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        plt.plot(degree_distribution_table.Degree, degree_distribution_table.pk, linestyle='', marker='x', color=color,label=label)
        plt.xlabel('Degree $k$')
        plt.ylabel('$P(k)$')
        plt.title('linear-linear scale', size=12)
        plt.legend()

        plt.subplot(1,2,2)
        plt.loglog(degree_distribution_table.Degree, degree_distribution_table.pk, linestyle='', marker='x', color=color,label=label)
        plt.xlabel('Degree $k$')
        plt.ylabel('$P(k)$')
        plt.title('log-log scale',size=12)
        plt.legend()

        plt.suptitle(title)

        plt.show()

    return degree_distribution_table




def plotDegreeRank(nodes, label='nodes', title='title', color='red', figsize=(12,5),save=False):
    """Plot the degree vs degree rank for a node set, return degree rank table. Plot both linear-linear scale and log-log scale

    Args:
        - nodes (dict) : dictionary object containing {node : degree} information for all nodes in set
        - label (str) : legend label for node set
        - color (str) : color of plot markers
        - save (bool) : flag to save figure (default False)
    Returns:
        - node_degree_table (pandas.core.frame.DataFrame) : python dataframe with degree and degree rank for nodes in node set
    """

    # Calculate rank of node degrees
    node_degree_table = pd.DataFrame.from_dict(nodes, orient='index').reset_index().rename(columns={'index' : 'congressperson', 0 : 'degree'})
    node_degree_table['degree_rank'] = node_degree_table['degree'].rank(ascending=False)
    node_degree_table.sort_values(by='degree', ascending=False).head(20)

    # Plot degree vs. degree rank
    with plt.style.context("ggplot"):
        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        plt.plot(node_degree_table['degree_rank'], node_degree_table['degree'], linestyle='', marker='x', color=color,label=label)
        plt.xlabel('Degree rank')
        plt.ylabel('Degree $k$')
        plt.title('linear-linear scale', size=12)
        plt.legend()

        plt.subplot(1,2,2)
        plt.loglog(node_degree_table['degree_rank'], node_degree_table['degree'], linestyle='', marker='x', color=color,label=label)
        plt.xlabel('Degree rank')
        plt.ylabel('Degree $k$')
        plt.title('log-log scale',size=12)
        plt.legend()

        plt.suptitle(title)

        plt.show()

    return node_degree_table



# ----------------------- BIPARTITE NETWORK FUNCTIONS ----------------------- #

def bipartiteSets(G):
    """Get left and right node sets within a bipartite network along with their degrees
    
    Args:
        - G (networkx.classes.graph.Graph) : networkx bipartite graph object
    Returns:
        - right_node_degrees (dict) : dictionary object of {node_name : degree} structure for right nodes in bipartite
        - left_node_degrees (dict) : dictionary object of {node_name : degree} structure for left nodes in bipartite
    """
    if nx.is_connected(G) == False:
        print('Network not connected - obtaining giant component')
        G = getGiantComponent(G)
    
    right_nodes, left_nodes = bipartite.sets(G)
    degree_dct = dict(G.degree)
    right_node_degrees = {key: degree_dct[key] for key in degree_dct if key in right_nodes}
    left_node_degrees = {key: degree_dct[key] for key in degree_dct if key in left_nodes}

    return right_node_degrees, left_node_degrees


def bipartiteNetworkSummary(G, left_label='left', right_label='right'):
    """Print summary statistics of bipartite network

    Args:
        - G (networkx.classes.graph.Graph) : networkx bipartite graph object
        - left_label (str) : label for left nodes in bipartite network
        - right_label (str) : label for right nodes in bipartite network
    """
    if nx.is_bipartite(G):

        left_node_degrees, right_node_degrees  = bipartiteSets(G)

        print('-' * 42)
        print('| {:<22} | {:<15} |'.format('Network Property', 'Value'))
        print('|' + '-' * 42 + '|')
        print('| {:<22} | {:<15} |'.format('#. of nodes', G.number_of_nodes()))
        print('| {:<22} | {:<15} |'.format('#. of edges', G.number_of_edges()))
        print('|' + '-' * 42 + '|')
        print('| {:<22} | {:<15} |'.format(f'#. of {left_label} nodes', len(left_node_degrees)))
        print('| {:<22} | {:<15} |'.format(f'Avg. {left_label} node degree', np.round(np.mean(list(left_node_degrees.values())), 2)))
        print('| {:<22} | {:<15} |'.format(f'Max {left_label} degree', max(left_node_degrees.values())))
        print('| {:<22} | {:<15} |'.format(f'Min {left_label} degree', min(left_node_degrees.values())))
        print('|' + '-' * 42 + '|')
        print('| {:<22} | {:<15} |'.format(f'#. of {right_label} nodes', len(right_node_degrees)))
        print('| {:<22} | {:<15} |'.format(f'Avg. {right_label} node degree', np.round(np.mean(list(right_node_degrees.values())), 2)))
        print('| {:<22} | {:<15} |'.format(f'Max {right_label} degree', max(right_node_degrees.values())))
        print('| {:<22} | {:<15} |'.format(f'Min {right_label} degree', min(right_node_degrees.values())))
        print('-' * 42)
    else:
        print('Network is not bipartite')