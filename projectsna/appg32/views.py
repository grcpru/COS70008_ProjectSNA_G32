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
    def generate_network_data(data, source_col='exporter_name', target_col='importer_name', attribute='value', weight_threshold=20):
        G = nx.DiGraph()
 
        if attribute in data.columns:
            data[attribute] = pd.to_numeric(data[attribute], errors='coerce')
 
        for _, row in data.iterrows():
            if attribute in data.columns and pd.notnull(row[attribute]):
                if row[attribute] >= weight_threshold:
                    G.add_edge(row[source_col], row[target_col], weight=row[attribute], trade_amount=row[attribute])
            else:
                G.add_edge(row[source_col], row[target_col])
 
        communities = community_louvain.best_partition(G.to_undirected(), weight='weight' if attribute in data.columns else None)
        num_communities = len(set(communities.values()))
 
        nodes = []
        edges = []
        cmap = plt.get_cmap('viridis')
 
        for node in G.nodes:
            color = mcolors.rgb2hex(cmap(communities[node] / num_communities))
            nodes.append({
                "id": node,
                "label": node,
                "title": f"Community: {communities[node]}",
                "group": communities[node],
                "color": color
            })
 
        for src, dst, attr in G.edges(data=True):
            if attribute in attr:
                title = f"Trade Amount: {attr['trade_amount']}"
                edges.append({
                    "from": src,
                    "to": dst,
                    "value": attr.get('weight', 1),
                    "title": title
                })
            else:
                edges.append({
                    "from": src,
                    "to": dst
                })
 
        grouped_nodes = {}
        for node in nodes:
            group = node['group']
            if group not in grouped_nodes:
                grouped_nodes[group] = []
            grouped_nodes[group].append(node)
 
        return nodes, edges, grouped_nodes
 
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    df = pd.read_csv(file_path)
 
    nodes, edges, grouped_nodes = generate_network_data(df)
 
    return render(request, "visualization_page.html", {
        "nodes_json": json.dumps(nodes),
        "edges_json": json.dumps(edges),
        "nodes": nodes,
        "grouped_nodes": grouped_nodes
    })

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

# Function to load and prepare the graph data
def load_and_prepare_graph(data_file_path):
    data = pd.read_csv(data_file_path)
    G = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr='value')
    return G

# Function to perform MCMC simulation
def mcmc_simulation(G, param1, param2, iterations):
    graphs = []
    for _ in range(iterations):
        G_copy = copy.deepcopy(G)
        # Simulate adding or removing edges
        if random.random() < 0.5:
            G_copy.add_edge(random.choice(list(G.nodes())), random.choice(list(G.nodes())))
        else:
            if G_copy.edges():
                edge_to_remove = random.choice(list(G_copy.edges()))
                G_copy.remove_edge(*edge_to_remove)
        graphs.append(G_copy)
    return graphs

# Function to fit ERGM model
def fit_ergm(G, coeff_samples=100, graph_samples=100):
    edge_coeffs = [random.normalvariate(0, 1)]
    triangle_coeffs = [random.normalvariate(0, 1)]
    probs = [0.1]  # Initial small probability to avoid division by zero later
    for _ in range(coeff_samples):
        new_edge_coeff = edge_coeffs[-1] + random.normalvariate(0, 0.1)
        new_triangle_coeff = triangle_coeffs[-1] + random.normalvariate(0, 0.1)
        graphs = mcmc_simulation(G, new_edge_coeff, new_triangle_coeff, graph_samples)
        new_prob = random.random()  # Simulate probability calculation
        probs.append(new_prob)
        edge_coeffs.append(new_edge_coeff)
        triangle_coeffs.append(new_triangle_coeff)

    # Normalize probabilities
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]

    # Calculate weighted averages
    weighted_edge_coeff = sum(p * e for p, e in zip(probs, edge_coeffs))
    weighted_triangle_coeff = sum(p * t for p, t in zip(probs, triangle_coeffs))

    best_idx = np.argmax(probs)
    best_edge_coeff = edge_coeffs[best_idx]
    best_triangle_coeff = triangle_coeffs[best_idx]

    return edge_coeffs, triangle_coeffs, best_edge_coeff, best_triangle_coeff, weighted_edge_coeff, weighted_triangle_coeff


# View function to analyze data
def analyze_data(request):
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    G = load_and_prepare_graph(file_path)
    edge_coeffs, triangle_coeffs, best_edge_coeff, best_triangle_coeff, weighted_edge_coeff, weighted_triangle_coeff = fit_ergm(G)

    # Line plot of coefficients
    plt.figure(figsize=(10, 5))
    plt.plot(edge_coeffs, label='Edge Coefficient')
    plt.plot(triangle_coeffs, label='Triangle Coefficient')
    plt.xlabel('Iteration')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficient Evolution Over Iterations')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    line_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Histograms
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].hist(edge_coeffs, bins=20, color='blue')
    axes[0].set_title('Edge Coefficient')
    axes[0].set_xlabel('Coefficient Value')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(triangle_coeffs, bins=20, color='blue')
    axes[1].set_title('Triangle Coefficient')
    axes[1].set_xlabel('Coefficient Value')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    hist_plot = base64.b64encode(buf.getvalue()).decode('utf-8')

    context = {
        'best_edge_coeff': best_edge_coeff,
        'best_triangle_coeff': best_triangle_coeff,
        'weighted_edge_coeff': weighted_edge_coeff,
        'weighted_triangle_coeff': weighted_triangle_coeff,
        'line_plot': line_plot,
        'hist_plot': hist_plot
    }
    return render(request, 'analysis_results.html', context)
# Example view function to plot results
def plot_ergm_results(request):
    # This view would involve generating plots and converting them to images for the web
    pass
