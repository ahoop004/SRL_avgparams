import xml.etree.ElementTree as ET
from collections import deque, defaultdict
from multiprocessing import Pool, cpu_count
import argparse

def parse_edges(file_path):
    """
    Parse the osm.net.xml file and extract valid edge IDs and their lengths, skipping internal edges.

    Args:
        file_path (str): Path to the osm.net.xml file.

    Returns:
        dict: A dictionary of valid edge IDs and their corresponding lengths.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    edges = {}

    for edge in root.findall("edge"):
        # Skip internal edges and edges with a "function" attribute
        if edge.get("function") is None:
            edge_id = edge.get("id")
            # Get the length of the first lane (assuming all lanes on the same edge have the same length)
            lane = edge.find("lane")
            length = float(lane.get("length")) if lane is not None else 0.0
            edges[edge_id] = length
    
    return edges

def parse_connections(file_path, valid_edges):
    """
    Parse the osm.net.xml file and extract connections between valid edges.

    Args:
        file_path (str): Path to the osm.net.xml file.
        valid_edges (dict): A dictionary of valid edge IDs and their corresponding lengths.

    Returns:
        dict: A dictionary where keys are edge IDs and values are lists of connected edge IDs.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    connections = defaultdict(list)

    for connection in root.findall("connection"):
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        
        # Only include connections where both from and to edges are in the valid edges set
        if from_edge in valid_edges and to_edge in valid_edges:
            connections[from_edge].append(to_edge)
    
    return connections

def find_route(connections, edge_lengths, start_edge, end_edge):
    """
    Find a route from the start edge to the end edge using Breadth-First Search (BFS)
    and calculate the total length of the route.

    Args:
        connections (dict): A dictionary where keys are edge IDs and values are lists of connected edge IDs.
        edge_lengths (dict): A dictionary of edge IDs and their corresponding lengths.
        start_edge (str): The starting edge ID.
        end_edge (str): The destination edge ID.

    Returns:
        tuple: A tuple containing the route as a list of edges and the total length of the route,
               or None if no route is found.
    """
    # BFS setup
    queue = deque([(start_edge, [start_edge], edge_lengths[start_edge])])  # Queue stores (current_edge, path, total_length)
    visited = set()

    while queue:
        current_edge, path, total_length = queue.popleft()
        
        # If we reached the destination, return the path and total length
        if current_edge == end_edge:
            return path, total_length
        
        # Mark the current edge as visited
        visited.add(current_edge)
        
        # Add connected edges to the queue
        for neighbor in connections.get(current_edge, []):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor], total_length + edge_lengths[neighbor]))
    
    return None  # No route found

def worker(args):
    """
    Worker function for multiprocessing to find routes for a batch of edge pairs.

    Args:
        args (tuple): Contains connections dict, edge_lengths dict, and a list of edge pairs.

    Returns:
        list: A list of results containing routes and lengths, or no route messages.
    """
    connections, edge_lengths, edge_pairs = args
    results = []

    for start_edge, end_edge in edge_pairs:
        if start_edge != end_edge:
            result = find_route(connections, edge_lengths, start_edge, end_edge)
            if result:
                route, total_length = result
                route_str = ' '.join(route)
                results.append(f"{start_edge} to {end_edge}: {route_str}, Length: {total_length:.2f}")
            else:
                results.append(f"No route found from {start_edge} to {end_edge}")
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Parse a SUMO network file and find routes between edges.")
    parser.add_argument('--input', required=True, help='Path to the input XML file (e.g., osm.net.xml)')
    parser.add_argument('--output', required=True, help='Path to the output file to save the routes')
    parser.add_argument('--processes', type=int, default=cpu_count(), help='Number of processes to use for parallelization')
    args = parser.parse_args()

    # Parse edges and connections from the input XML file
    edge_lengths = parse_edges(args.input)
    connections = parse_connections(args.input, edge_lengths)

    # Generate all pairs of edges
    edges = list(edge_lengths.keys())
    edge_pairs = [(start_edge, end_edge) for start_edge in edges for end_edge in edges if start_edge != end_edge]

    # Split the edge pairs into chunks for each process
    num_processes = args.processes
    chunk_size = len(edge_pairs) // num_processes
    chunks = [edge_pairs[i:i + chunk_size] for i in range(0, len(edge_pairs), chunk_size)]

    # Create a pool of workers and distribute the work
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, [(connections, edge_lengths, chunk) for chunk in chunks])

    # Flatten the results and write to the output file
    with open(args.output, 'w') as f:
        for result in results:
            for line in result:
                f.write(line + '\n')

if __name__ == "__main__":
    main()
