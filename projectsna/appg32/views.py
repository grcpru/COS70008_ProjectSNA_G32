from django.http import HttpResponse
import os
import logging
import pandas as pd
import numpy as np
import dtale
import dtale.views
from django.shortcuts import render, HttpResponseRedirect
from django.http import HttpResponse
from .forms import DatasetUploadForm
from django.conf import settings
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from community import community_louvain
import plotly.graph_objs as go
import plotly.offline as pyo
import random  
import copy
import json
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from django.template.loader import get_template


# Switch to 'Agg' backend for matplotlib
plt.switch_backend('Agg')

def landing_page(request):
    return render(request, 'landing_page.html')

def upload_data(request):
    if request.method == 'GET':
        return render(request, 'upload_form.html')
    elif request.method == 'POST':
        return HttpResponseRedirect('/success/')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_file = form.cleaned_data['dataset']
            file_path = os.path.join(settings.MEDIA_ROOT, dataset_file.name)
            with open(file_path, 'wb') as destination:
                for chunk in dataset_file.chunks():
                    destination.write(chunk)
            return HttpResponseRedirect('/dataset-success/')
        else:
            return render(request, 'upload_dataset_form.html', {'form': form})
    else:
        form = DatasetUploadForm()
        return render(request, 'upload_dataset_form.html', {'form': form})

def explore_datasets(request):
    # Path to the media folder
    media_folder = 'media'
    # List all files in the media folder
    files = os.listdir(media_folder)
    return render(request, 'explore_datasets.html', {'files': files})

def explore_file(request, file):
    # Path to the selected file in the media folder
    file_path = os.path.join('media', file)
    # Read the selected file as a pandas DataFrame
    df = pd.read_csv(file_path)
    # Start a D-Tale web server and get the URL
    dtale_instance = dtale.show(df)
    dtale_url = dtale_instance.main_url()
    return render(request, 'explore_file.html', {'dtale_url': dtale_url})

def upload_success(request):
    return render(request, 'upload_success.html')

def analyze_data(request):
    # Define the path to the uploaded file
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    
    # Read the uploaded file as a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Create the network graph from the DataFrame
    G_df = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr='value')
    
    # Fit the ERGM model if needed
    # edge_coeff, triangle_coeff, _ = fit_ergm(G_df)
    
    # Generate positions using a layout
    pos = nx.spring_layout(G_df)
    
    # Extract node positions
    x_nodes = [pos[k][0] for k in pos]
    y_nodes = [pos[k][1] for k in pos]
    
    # Create traces for edges
    edge_trace = go.Scatter(
        x=sum([[pos[edge[0]][0], pos[edge[1]][0], None] for edge in G_df.edges()], []),
        y=sum([[pos[edge[0]][1], pos[edge[1]][1], None] for edge in G_df.edges()], []),
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create trace for nodes
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
            size=10,
            line=dict(width=2)
        ),
        text=[f"Node: {node}" for node in G_df.nodes()],
        hoverinfo='text'
    )
    
    # Update node colors based on the number of connections (node degree)
    node_adjacencies = [len(G_df.adj[node]) for node in G_df.nodes()]
    node_trace.marker.color = node_adjacencies
    
    # Set up layout for the graph
    fig_layout = go.Layout(
        title='<br>Interactive Network Graph with Plotly and NetworkX',
        titlefont=dict(size=16),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=30),
        annotations=[dict(
            text="Network Visualization",
            showarrow=False,
            xref="paper", yref="paper"
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=fig_layout)
    
    # Render the graph and pass it to the template
    div = pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    return render(request, 'network_graph.html', {'graph_div': div})

def network_graph(request):
    # Render the network graph template
    return render(request, 'network_graph.html')

def network_view(request):
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    df = pd.read_csv(file_path)

    def generate_network_graph(df, source_col='exporter_name', target_col='importer_name', attribute='value', weight_threshold=20):
        G = nx.DiGraph()

        # Ensure numeric conversion for the attribute, if it exists
        if attribute in df.columns:
            df[attribute] = pd.to_numeric(df[attribute], errors='coerce')

        # Add edges to the graph based on the weight threshold
        for _, row in df.iterrows():
            if attribute in df.columns and pd.notnull(row[attribute]):
                if row[attribute] >= weight_threshold:
                    G.add_edge(row[source_col], row[target_col], weight=row[attribute], trade_amount=row[attribute])
            else:
                G.add_edge(row[source_col], row[target_col])  # Add edges without weight if attribute is missing or below threshold

        # Apply community detection to the whole graph
        communities = community_louvain.best_partition(G.to_undirected(), weight='weight' if attribute in df.columns else None)
        num_communities = len(set(communities.values()))

        # Initialize Pyvis Network with a white background, filter menu, and select menu
        net = Network(height="750px", width="100%", bgcolor="white", font_color="black", filter_menu=True, select_menu=True)

        # Adding nodes with color by community
        cmap = plt.get_cmap('viridis')
        for node in G.nodes:
            color = mcolors.rgb2hex(cmap(communities[node] / num_communities))
            net.add_node(node, title=f"Community: {communities[node]}", group=communities[node], node_color=color)

        # Adding edges with detailed information
        for src, dst, attr in G.edges(data=True):
            if attribute in attr:
                title = f"{attribute.capitalize()}: {attr['trade_amount']}"
                net.add_edge(src, dst, value=attr['weight'], title=title)
            else:
                net.add_edge(src, dst)  # Add edge without title if attribute is missing

        return net

    net = generate_network_graph(df)

    # Extract nodes and edges data from the net object
    nodes = [{"id": node["id"], "label": node["label"], "color": node["node_color"], "group": node["group"]} for node in net.nodes]
    edges = [{"from": edge["from"], "to": edge["to"], "value": edge.get("value", 1), "title": edge.get("title", "")} for edge in net.edges]

    graph_data = {
        "nodes": nodes,
        "edges": edges
    }

    communities = list(set([node["group"] for node in nodes]))
    node_list = [{"id": node["id"], "label": node["label"]} for node in nodes]

    return render(request, "visualization_page.html", {"graph_data": json.dumps(graph_data), "communities": communities, "nodes": node_list})

def calculate_network_statistics(dataframe):
    G = nx.from_pandas_edgelist(dataframe, source='exporter_name', target='importer_name', edge_attr=True, create_using=nx.DiGraph())

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / float(num_nodes)
    network_density = nx.density(G)
    clustering_coefficient = nx.average_clustering(G.to_undirected())
    
    in_degrees = np.array(list(dict(G.in_degree()).values()))
    out_degrees = np.array(list(dict(G.out_degree()).values()))
    betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))

    degrees = [d for n, d in G.degree()]
    degree_count = pd.Series(degrees).value_counts().sort_index()

    # Create a plot as a string buffer
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.bar(degree_count.index, degree_count.values, width=0.80, color='b')
    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')

    single_stats_df = pd.DataFrame({
        'Statistic': ['Number of Nodes', 'Number of Edges', 'Average Degree', 'Network Density', 'Average Clustering Coefficient'],
        'Value': [num_nodes, num_edges, avg_degree, network_density, clustering_coefficient]
    })

    multi_stats_df = pd.DataFrame({
        'Statistic': ['In-Degree', 'Out-Degree', 'Betweenness Centrality'],
        'Total': [np.sum(in_degrees), np.sum(out_degrees), np.nan],
        'Average': [np.mean(in_degrees), np.mean(out_degrees), np.mean(betweenness_centrality)],
        'Max': [np.max(in_degrees), np.max(out_degrees), np.max(betweenness_centrality)],
        'Standard Deviation': [np.std(in_degrees), np.std(out_degrees), np.std(betweenness_centrality)]
    })

    return single_stats_df, multi_stats_df, graph

def network_statistics(request):
    # Load your data
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    df = pd.read_csv(file_path)

    # Calculate statistics and generate graph
    single_stats, multi_stats, graph = calculate_network_statistics(df)

    # Convert DataFrames to HTML for rendering in the template
    single_stats_html = single_stats.to_html(classes='table table-striped', index=False)
    multi_stats_html = multi_stats.to_html(classes='table table-striped', index=False)

    context = {
        'single_stats': single_stats_html,
        'multi_stats': multi_stats_html,
        'graph': graph  # graph is a base64-encoded PNG image
    }
    return render(request, 'network_statistics.html', context)

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

def data_prep_model():
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
        data = pd.read_csv(file_path)
        edges = data[['exporter_name', 'importer_name', 'value']].copy()
        G_df = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr='value')

        #G_df = nx.from_pandas_edgelist(edges, 'exporter_name', 'importer_name', edge_attr=True)
        #G_df = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr=['value'])

        return G_df
    except Exception as e:
        print(f"Error in data_prep_model: {e}")
        return None
    
def fit_ergm(G, coeff_samples=100, graph_samples=1000, return_all=False):
    edge_coeffs = []
    triangle_coeffs = []
    probs = []
    for _ in range(coeff_samples):
        edge_coeff = random.normalvariate(0, 1)
        triangle_coeff = random.normalvariate(0, 1)
        graphs = mcmc(G, edge_coeff, triangle_coeff, graph_samples)
        sum_weight = sum_weights(graphs, edge_coeff, triangle_coeff)
        if sum_weight > 0:  # Make sure sum_weight is not zero or negative
            prob = compute_weight(G, edge_coeff, triangle_coeff) / sum_weight
            edge_coeffs.append(edge_coeff)
            triangle_coeffs.append(triangle_coeff)
            probs.append(prob)

    if not probs:  # Check if the list is empty
        return None  # or return a default like (0, 0, 0)

    best_index = np.argmax(probs)
    if return_all:
        return (edge_coeffs, triangle_coeffs, probs)
    else:
        return (edge_coeffs[best_index], triangle_coeffs[best_index], probs[best_index])

def plot_ergm_results(edge_coeffs, triangle_coeffs, probs):
    if not edge_coeffs or not triangle_coeffs or not probs:
        return None  # Return None if any list is empty

    try:
        i = np.argmax(probs)
        max_prob = max(probs)
        best_edge_coeff = edge_coeffs[i]
        best_triangle_coeff = triangle_coeffs[i]

        fig1 = plt.figure(figsize=(8, 4))
        ax = fig1.add_subplot(111)
        p1, = ax.plot(edge_coeffs)
        p2, = ax.plot(triangle_coeffs)
        ax.set_ylabel("Coefficient Value")
        ax.set_xlabel("Iteration")
        ax.legend([p1, p2], ["Edge Coefficient", "Triangle Coefficient"])

        weighted_edge_coeffs = np.array(edge_coeffs[1:]) * np.array(probs[1:])
        weighted_edge_coeff = np.sum(weighted_edge_coeffs) / np.sum(probs[1:])

        weighted_tri_coeffs = np.array(triangle_coeffs[1:]) * np.array(probs[1:])
        weighted_tri_coeff = np.sum(weighted_tri_coeffs) / np.sum(probs[1:])

        fig2 = plt.figure(figsize=(12, 4))
        ax1 = fig2.add_subplot(121)
        ax1.hist(edge_coeffs[1:], weights=-np.log(probs[1:]))
        ax1.set_title("Edge Coefficient")
        ax1.set_xlabel("Coefficient Value")

        ax2 = fig2.add_subplot(122)
        ax2.hist(triangle_coeffs[1:], weights=-np.log(probs[1:]))
        ax2.set_title("Triangle Coefficient")
        ax2.set_xlabel("Coefficient Value")

        plt.close('all')  # Ensure all figures are closed after creation
        return {
            'max_prob': max_prob,
            'best_edge_coeff': best_edge_coeff,
            'best_triangle_coeff': best_triangle_coeff,
            'weighted_edge_coeff': compute_weighted_coeff(edge_coeffs, probs),
            'weighted_tri_coeff': compute_weighted_coeff(triangle_coeffs, probs),
            'fig1': fig1,
            'fig2': fig2
        }
    except Exception as e:
        print(f"Error in plot_ergm_results: {e}")
        return None


# Configure logging
logger = logging.getLogger(__name__)

def save_plot(fig, filename):
    try:
        path = os.path.join(settings.PLOT_STORAGE_DIR, filename)
        fig.savefig(path)
        plt.close(fig)  # Close the plot explicitly after saving
        return path
    except Exception as e:
        print(f"Failed to save the plot: {e}")
        return None

import logging

logger = logging.getLogger(__name__)

def analyze_dataset(request, filename):
    try:
        # Construct the full file path
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        # Attempt to read the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        # Convert DataFrame to a NetworkX graph
        G_df = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr='value')

        # Fit the ERGM model
        results = fit_ergm(G_df, 100, 10, True)
        if not results or not all(results):
            return HttpResponse("Model fitting failed to produce results or produced empty results.", status=500)

        edge_coeffs, triangle_coeffs, probs = results
        if not edge_coeffs or not triangle_coeffs or not probs:
            return HttpResponse("Model fitting produced empty coefficients or probabilities.", status=500)

        # Generate plots based on the ERGM results
        plot_results = plot_ergm_results(edge_coeffs, triangle_coeffs, probs)
        if not plot_results:
            return HttpResponse("Failed to generate plots.", status=500)

        # Save plots to files
        fig1_path = save_plot(plot_results['fig1'], 'ergm_plot1.png')
        fig2_path = save_plot(plot_results['fig2'], 'ergm_plot2.png')
        if not fig1_path or not fig2_path:
            return HttpResponse("Failed to save plot images.", status=500)

        # Prepare context for the template rendering
        context = {
            'max_prob': plot_results['max_prob'],
            'best_edge_coeff': plot_results['best_edge_coeff'],
            'best_triangle_coeff': plot_results['best_triangle_coeff'],
            'weighted_edge_coeff': plot_results['weighted_edge_coeff'],
            'weighted_tri_coeff': plot_results['weighted_tri_coeff'],
            'fig1_path': fig1_path,
            'fig2_path': fig2_path
        }

        # Render the results template with the context
        return render(request, 'analysis_results.html', context)

    except FileNotFoundError:
        return HttpResponse("The specified file could not be found.", status=404)
    except pd.errors.EmptyDataError:
        return HttpResponse("No data found in the file.", status=400)
    except Exception as e:
        return HttpResponse(f"An error occurred: {e}", status=500)
