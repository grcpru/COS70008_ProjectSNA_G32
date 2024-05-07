import os
import pandas as pd
import dtale
import dtale.views
from django.shortcuts import render, HttpResponseRedirect
from .forms import DatasetUploadForm
from django.conf import settings
import networkx as nx
import plotly.graph_objs as go
import plotly.offline as pyo
import random  
import copy



def landing_page(request):
    return render(request, 'landing_page.html')

def upload_data(request):
    if request.method == 'GET':
        return render(request, 'upload_form.html')
    elif request.method == 'POST':
        return HttpResponseRedirect('/success/')

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
        return best_edge_coeff, best_triangle_coeff, best_p  # Modified line
    else:
        return edge_coeffs, triangle_coeffs, probs

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


def data_prep_model(data_file_path):
    data = pd.read_csv(data_file_path, index_col=0)
    edges = data[['exporter_name','importer_name','value']].copy().reset_index(drop=True)
    G_df = nx.from_pandas_edgelist(edges, 'exporter_name', 'importer_name', edge_attr='value')
    return G_df
