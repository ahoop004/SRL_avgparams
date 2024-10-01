import xml.etree.ElementTree as ET
import sumolib
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


class NetParser:
    """
    NetParser class for parsing network files and retrieving information from SUMO simulations.

    Attributes:
        sumocfg (str): Path to the SUMO configuration file.
    """

    def __init__(self, sumocfg) -> None:
        """
        Initialize the NetParser object with a specific SUMO configuration file.

        Args:
            sumocfg (str): Path to the SUMO configuration file.
        """
        self.sumocfg = sumocfg

    def parse_net_files(self):
        """
        Get the network file from the SUMO configuration file.

        Returns:
            str: Path to the network file extracted from the SUMO configuration.
        """
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for infile in root.findall("input"):
            for network in infile.findall("net-file"):
                return str(network.get("value"))

    def _clean_path(self):
        """
        Clean the file path for the network file.

        Returns:
            sumolib.net.Net: The network object after reading the network file.
        """
        net_file = self.parse_net_files()
        path_b = "/".join(self.sumocfg.rsplit("/")[:-1])
        return sumolib.net.readNet(f"{path_b}/{net_file}")

    def get_edges_info(self):
        """
        Get a list of edges that allow passenger vehicles.

        Returns:
            list: List of edges allowing passenger vehicles.
        """
        net = self._clean_path()
        return [edge for edge in net.getEdges() if edge.allows("passenger")]

    def get_edge_pos_dic(self):
        """
        Get a dictionary of edge IDs and their XY coordinates at the center.

        Returns:
            dict: Dictionary of edge IDs and their center XY coordinates.
        """
        net = self._clean_path()
        return {
            edge.getID(): (
                (edge.getShape()[0][0] + edge.getShape()[1][0]) / 2,
                (edge.getShape()[0][1] + edge.getShape()[1][1]) / 2
            )
            for edge in net.getEdges()
        }

    def get_out_dic(self):
        """
        Get a dictionary of edges and their connecting edges.

        Returns:
            dict: Dictionary of edges and their respective connecting edges.
        """
        net = self._clean_path()
        out_dict = {}
        for edge in net.getEdges():
            if edge.allows("passenger"):
                out_dict[edge.getID()] = {
                    conn.getDirection(): out_edge.getID()
                    for out_edge in edge.getOutgoing()
                    if out_edge.allows("passenger")
                    for conn in edge.getConnections(out_edge)
                }
        return out_dict

    def get_edge_index(self):
        """
        Get an indexed dictionary of edge IDs.

        Returns:
            dict: Indexed dictionary of edge IDs.
        """
        net = self._clean_path()
        return {edge.getID(): idx for idx, edge in enumerate(net.getEdges())}

    def get_length_dic(self):
        """
        Get a dictionary of edge IDs and their lengths.

        Returns:
            dict: Dictionary of edge IDs and their lengths.
        """
        net = self._clean_path()
        return {edge.getID(): edge.getLength() for edge in net.getEdges()}

    def get_route_edges(self):
        """
        Get a list of edge IDs from a specific route file.

        Returns:
            list: List of edge IDs from the specified route.
        """
        return [
            route.edges.split()
            for route in sumolib.xml.parse_fast("Experiments/balt1/Nets/osm_pt.rou.xml", 'route', ['id', 'edges'])
            if 'bus' in route.id
        ][0]  # assuming only one match

    def get_max_manhattan(self):
        """
        Calculate the maximum Manhattan distance between any two edges in the network.

        Returns:
            float: Maximum Manhattan distance.
        """
        positions = list(self.get_edge_pos_dic().values())
        V = [x + y for x, y in positions]
        V1 = [x - y for x, y in positions]
        V.sort()
        V1.sort()
        return max(V[-1] - V[0], V1[-1] - V1[0])

    def net_minmax(self):
        """
        Get net minmax xy coordinates for scaling input.

        Returns:
            tuple: Minmax xy coordinates.
        """
        net = self._clean_path()
        return sumolib.net.Net.getBBoxXY(net)

    def get_junctions(self):
        """
        Retrieve junctions and their internal edges from the network.

        Returns:
            set: Set of junctions and internal edges.
        """
        net = self._clean_path()
        junctions = {
            junction.getID() for junction in net.getNodes() if junction.getType() != "internal"
        }
        junctions.update(
            edge.getID() for edge in net.getEdges() if edge.getID().startswith(":")
        )
        return junctions
