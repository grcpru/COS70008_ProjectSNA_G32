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
from .subdirectory.ergm import compute_weight, permute_graph, mcmc, sum_weights, fit_ergm
from .subdirectory import ergm


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

def analyze_data(request):
    # Define the path to the uploaded file
    file_path = os.path.join(settings.MEDIA_ROOT, 'trade_data_rice.csv')
    
    # Read the uploaded file as a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Create the network graph from the DataFrame
    G_df = nx.from_pandas_edgelist(data, 'exporter_name', 'importer_name', edge_attr='value')
    
    # Fit the ERGM model if needed
    # edge_coeff, triangle_coeff, _ = ergm.fit_ergm(G_df)
    
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
