import numpy as np
import pandas as pd
import random
import networkx as nx
import copy

def compute_weight(G, edge_coeff, tri_coeff):
    '''
    Compute the probability weight on graph G
    '''
    edge_count = len(G.edges())
    triangles = sum(nx.triangles(G).values())
    return np.exp(edge_count * edge_coeff + triangles * tri_coeff)

def permute_graph(G):
    '''
    Return a new graph with an edge randomly added or subtracted from G
    '''
    G1 = copy.deepcopy(G)
    d = nx.density(G1)
    r = random.random()
    if (r < 0.5 or d == 0) and d != 1:
        # Add an edge
        nodes = G.nodes()
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        G1.add_edge(n1, n2)
    else:
        # Remove an edge
        n1, n2 = random.choice(G1.edges())
        G1.remove_edge(n1, n2)
    return G1

def mcmc(G, edge_coeff, triangle_coeff, n):
    '''
    Use MCMC to generate a sample of networks from an ERG distribution.

    Args:
        G: The observed network, to seed the graph with
        edge_coeff: The coefficient on the number of edges
        triangle_coeff: The coefficient on number of triangles
        n: The number of samples to generate
    Returns:
        A list of graph objects
    '''

    v = len(G) # number of nodes in G
    p = nx.density(G) # Probability of a random edge existing
    current_graph = nx.erdos_renyi_graph(v, p) # Random graph
    current_w = compute_weight(G, edge_coeff, triangle_coeff)
    graphs = []
    while len(graphs) < n:
        new_graph = permute_graph(current_graph)
        new_w = compute_weight(new_graph, edge_coeff, triangle_coeff)
        if new_w > current_w or random.random() < (new_w/current_w):
            graphs.append(new_graph)
            current_w = new_w
    return graphs

def sum_weights(graphs, edge_coeff, tri_coeff):
    '''
    Sum the probability weights on every graph in graphs
    '''
    total = 0.0
    for g in graphs:
        total += compute_weight(g, edge_coeff, tri_coeff)
    return total

def fit_ergm(G, coeff_samples=100, graph_samples=1000, return_all=False):
    '''
    Use MCMC to sample possible coefficients, and return the best fits.

    Args:
        G: The observed graph to fit
        coeff_samples: The number of coefficient combinations to sample
        graph_samples: The number of graphs to sample for each set of coeffs
        return_all: If True, return all sampled values. Otherwise, only best.
    Returns:
        If return_all=False, returns a tuple of values,
            (best_edge_coeff, best_triangle_coeff, best_p)
        where p is the estimated probability of observing the graph G with
        the fitted parameters.

        Otherwise, return a tuple of lists:
            (edge_coeffs, triangle_coeffs, probs)
    '''
    edge_coeffs = [0]
    triangle_coeffs = [0]
    probs = [None]

    while len(probs) < coeff_samples:
        # Make the jump size larger early on, and smaller toward the end
        w = coeff_samples/50.0
        s = np.sqrt(w/len(probs))
        # Pick new coefficients to try:
        edge_coeff = edge_coeffs[-1] +  random.normalvariate(0, s)
        triangle_coeff = triangle_coeffs[-1] + random.normalvariate(0, s)
        # Check how likely the observed graph is under this distribution:
        graphs = mcmc(G, edge_coeff, triangle_coeff, graph_samples)
        sum_weight = sum_weights(graphs, edge_coeff, triangle_coeff)
        p = compute_weight(G, edge_coeff, triangle_coeff) / sum_weight
        # Decide whether to accept the jump:
        if p > probs[-1] or random.random() < (p / probs[-1]):
            edge_coeffs.append(edge_coeff)
            triangle_coeffs.append(triangle_coeff)
            probs.append(p)
        else:
            edge_coeffs.append(edge_coeffs[-1])
            triangle_coeffs.append(triangle_coeffs[-1])
            probs.append(probs[1])
    # Return either the best values, or all of them:
    if not return_all:
        i = np.argmax(probs)
        best_p = probs[i]
        best_edge_coeff = edge_coeffs[i]
        best_triangle_coeff = triangle_coeffs[i]
        return (best_edge_coeff, best_triangle_coeff, best_p)
    else:
        return (edge_coeffs, triangle_coeffs, probs)
