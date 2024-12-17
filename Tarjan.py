import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, vertices):
        """
        Initialize a graph object.
        
        Parameters:
        vertices (int): Number of vertices in the graph.
        """
        self.V = vertices  # Number of vertices
        self.graph = [[] for _ in range(vertices)]  # Adjacency list representation of the graph
        self.time = 0  # Timer to keep track of discovery times

    def addEdge(self, u, v):
        """
        Add a directed edge from vertex u to vertex v.
        
        Parameters:
        u (int): Start vertex of the edge.
        v (int): End vertex of the edge.
        """
        self.graph[u].append(v)

    def tarjansSCC(self):
        """
        Perform Tarjan's algorithm to find all strongly connected components (SCCs) in the graph.
        
        Returns:
        list: A list of SCCs, where each SCC is represented as a list of vertices.
        """
        # Initialize auxiliary data structures
        disc = [-1] * self.V  # Discovery times for vertices (-1 means unvisited)
        low = [-1] * self.V  # Lowest discovery time reachable from each vertex
        stack_member = [False] * self.V  # Boolean array to track stack membership
        st = []  # Stack to store the vertices of the current DFS path
        scc_list = []  # List to store all SCCs

        # Recursive utility function for Tarjan's algorithm
        def SCCUtil(u):
            """
            Perform depth-first search (DFS) to find SCCs starting from vertex u.
            
            Parameters:
            u (int): Current vertex being processed.
            """
            nonlocal scc_list
            
            # Initialize discovery and low values for u
            disc[u] = low[u] = self.time
            self.time += 1
            st.append(u)  # Push u onto the stack
            stack_member[u] = True  # Mark u as being on the stack

            # Explore all adjacent vertices of u
            for v in self.graph[u]:
                if disc[v] == -1:  # If v is not visited
                    SCCUtil(v)  # Recur for v
                    low[u] = min(low[u], low[v])  # Update low value of u based on v
                elif stack_member[v]:  # If v is in the stack, it's a back edge
                    low[u] = min(low[u], disc[v])

            # If u is a root node, pop all nodes in the SCC
            if low[u] == disc[u]:
                scc = []
                while True:
                    w = st.pop()  # Remove a vertex from the stack
                    stack_member[w] = False
                    scc.append(w)  # Add it to the current SCC
                    if w == u:
                        break
                scc_list.append(scc)  # Add the SCC to the result list

        # Call SCCUtil for all unvisited vertices
        for i in range(self.V):
            if disc[i] == -1:
                SCCUtil(i)

        return scc_list

def convert_networkx_to_graph(nx_graph):
    """
    Convert a NetworkX graph to a custom Graph object suitable for Tarjan's algorithm.
    
    Parameters:
    nx_graph (networkx.DiGraph): The NetworkX directed graph.

    Returns:
    tuple: A tuple containing the custom Graph object and a mapping from original nodes to integer indices.
    """
    # Map nodes of the NetworkX graph to integers
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    num_vertices = len(node_mapping)  # Total number of vertices in the graph

    # Initialize the custom Graph object
    g = Graph(num_vertices)

    # Add edges to the custom graph using the mapped indices
    for u, v in nx_graph.edges():
        g.addEdge(node_mapping[u], node_mapping[v])

    return g, node_mapping

# Main code execution
file_path = r"C:\Users\zadeboye\work\Wiki-Vote.txt"  # Path to the input data file
G = nx.DiGraph()  # Create a directed graph using NetworkX

# Read the input file and populate the graph
with open(file_path, "r") as file:
    for line in file:
        if line.startswith("#"):
            continue  # Skip comment lines
        from_node, to_node = map(int, line.strip().split())  # Parse edge endpoints
        G.add_edge(from_node, to_node)  # Add the edge to the graph

# Print basic information about the graph
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Select a subgraph for visualization and analysis (e.g., the first 500 nodes)
subgraph_nodes = list(G.nodes)[:200]  # Select the first 200 nodes
subgraph = G.subgraph(subgraph_nodes)  # Create a subgraph containing these nodes

# Convert the subgraph to the custom Graph object
tarjan_graph, mapping = convert_networkx_to_graph(subgraph)

# Perform Tarjan's algorithm to find SCCs
sccs = tarjan_graph.tarjansSCC()
print("\nStrongly Connected Components:")
for i, scc in enumerate(sccs, 1):
    print(f"SCC {i}: {scc}")

# Visualize the subgraph using NetworkX
plt.figure(figsize=(12, 8))
nx.draw(
    subgraph,
    with_labels=True,  # Display node labels
    node_size=100,  # Size of the nodes
    font_size=3,  # Font size for labels
    arrowstyle="->",  # Style of arrows for directed edges
    arrowsize=10,  # Size of arrows
)
plt.title("Subgraph Visualization (First 500 Nodes)")  # Set plot title
plt.show()

# Find the largest SCC
largest_scc = max(sccs, key=len)  # Find the SCC with the maximum size
largest_scc_length = len(largest_scc)  # Get its size

# Print the results
print(f"\nThe largest SCC has {largest_scc_length} nodes.")
print(f"Nodes in the largest SCC: {largest_scc}")
