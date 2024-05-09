function renderNetworkGraph(graphData) {
    var container = document.getElementById("network-graph");
    var data = {
        nodes: new vis.DataSet(graphData.nodes),
        edges: new vis.DataSet(graphData.edges)
    };

    var options = {
        height: "600px",
        width: "100%",
        nodes: {
            shape: "dot",
            size: 10
        },
        edges: {
            width: 1, // Set a base width for edges
            edgeScaling: {
                enabled: true,
                min: 0.1, // Set the minimum edge width
                max: 2, // Set the maximum edge width
                value: 0.2 // Scale the edges to this value
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            selectConnectedEdges: false
        },
        layout: {
            improvedLayout: true, // This will improve the positioning of the nodes
            hierarchical: {
                enabled: false // Disable the hierarchicalLayout
            },
            clusterThreshold: 10, // Adjust this value to control node clustering
            springLength: 15, // Adjust the spacing between nodes
            springConstant: 0.001 // Adjust the repulsion between nodes
        },
        physics: {
            repulsion: {
                centralGravity: 0.2, // Adjust the centralGravity value as needed
                nodeDistance: 200 // Increase the nodeDistance to create more space between nodes
            }
        }    
    };

    var network = new vis.Network(container, data, options);

    // Event for selecting a node
    network.on("selectNode", function(params) {
        var selectedNode = data.nodes.get(params.nodes[0]);
        console.log("Selected node:", selectedNode);
        // Add your logic here to display information about the selected node
    });

    // Event for deselecting a node
    network.on("deselectNode", function(params) {
        console.log("Node deselected");
        // Add your logic here to clear any information displayed for the previously selected node
    });

    // Function to filter the network by community
    var filterByCommunity = function(communityId) {
        var nodes = data.nodes.map(function(node) {
            node.hidden = node.group !== communityId;
            return node;
        });
        var edges = data.edges.map(function(edge) {
            var sourceNode = data.nodes.get(edge.from);
            var targetNode = data.nodes.get(edge.to);
            edge.hidden = sourceNode.group !== communityId || targetNode.group !== communityId;
            return edge;
        });
        data.nodes.update(nodes);
        data.edges.update(edges);
    };

    // Event for hovering over an edge
    network.on("hoverEdge", function (params) {
        var edgeData = data.edges.get(params.edge);
        var weightValue = edgeData.value || edgeData.weight || 1; // Fallback to 1 if no value or weight is provided

        // Update the options to display the weight value on hover
        network.setOptions({
            edges: {
                hoverWidth: function (edgeWidth) {
                    return Math.max(edgeWidth * 1.5, edgeWidth + 5);
                },
                hover: {
                    toArrowCalculation: function () {
                        return {
                            enabled: true,
                            type: 'arrow',
                            scaleFactor: 2
                        };
                    },
                    font: {
                        color: '#000000',
                        size: 14,
                        face: 'arial'
                    },
                    label: function () {
                        return weightValue.toString();
                    }
                }
            }
        });
    });

    // Function to highlight a node and its connections
    var highlightNode = function(nodeId) {
        var connectedNodes = new Set();
        var connectedEdges = [];

        // Find the edges connected to the selected node
        data.edges.forEach(function(edge) {
            if (edge.from === nodeId || edge.to === nodeId) {
                connectedEdges.push(edge.id);
                connectedNodes.add(edge.from);
                connectedNodes.add(edge.to);
            }
        });

        // Show the selected node, its connected nodes, and their edges
        data.nodes.update(data.nodes.map(function(node) {
            node.hidden = !connectedNodes.has(node.id) && node.id !== nodeId;
            return node;
        }));
        data.edges.update(data.edges.map(function(edge) {
            edge.hidden = !connectedEdges.includes(edge.id);
            return edge;
        }));
    };

    // Add an event listener for community selection
    var communitySelect = document.getElementById("community-select");
    communitySelect.addEventListener("change", function() {
        var selectedCommunity = this.value;
        if (selectedCommunity === "all") {
            data.nodes.update(data.nodes.map(function(node) {
                node.hidden = false;
                return node;
            }));
            data.edges.update(data.edges.map(function(edge) {
                edge.hidden = false;
                return edge;
            }));
        } else {
            filterByCommunity(parseInt(selectedCommunity));
        }
    });

    // Add an event listener for node selection
    var nodeSelect = document.getElementById("node-select");
    nodeSelect.addEventListener("change", function() {
        var selectedNodeId = this.value;
        if (selectedNodeId) {
            highlightNode(selectedNodeId);
        } else {
            // Reset node and edge visibility to default
            data.nodes.update(data.nodes.map(function(node) {
                node.hidden = false;
                return node;
            }));
            data.edges.update(data.edges.map(function(edge) {
                edge.hidden = false;
                return edge;
            }));
        }
    });
}