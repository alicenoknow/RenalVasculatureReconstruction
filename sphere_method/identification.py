"""This module contains functions necessary for the identification of vessel branches."""
# pylint: disable=import-error


from queue import SimpleQueue

import numpy as np
import networkx as nx

from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import euclidean
from skimage.draw import circle_perimeter, disk
from skimage.feature import peak_local_max


PROJECTION_ANGLES = {
    (45, 0): ["5", "6", "13", "9", "12a"],
    (-45, 0): ["5", "6", "13", "9", "12a"],
    (45, 20): ["5", "15"],
    (45, 20): ["5", "13"],
    (30, -30): ["5", "6", "13", "9", "12a"],
    (-30, -30): ["5", "13", "12a", "12b"],
    (-30, 30): ["5", "6", "9", "10", "10"],
}


def get_starting_points(img):
    """Get starting points for centerline detection algorithm. The starting point is selected
    as the maximum value of the euclidean distance transform (EDT) image of the segmented vessel.

    Args:
        img (np.array): EDT image of the segmented vessel.

    Returns:
        tuple: Starting point for centerline detection algorithm.
    """
    # initialize helper image
    helper = np.zeros_like(img)
    # find local maxima on EDT image.
    starting_points = peak_local_max(img)
    # get coordinates of local maxima
    helper[tuple(starting_points.T)] = 1
    starting_points = list(zip(*np.nonzero(helper)))
    # sort starting points by vessel diameter.
    starting_points.sort(key=lambda x: img[x])
    return starting_points


def get_points_of_interest(point, radius, visited, edt_img):
    """Get list of points that can be on the vessel centerline. The method utilizes local maxima
    to find the centerline on the basis of the EDT image of the vessel. The returned points are
    sorted by the EDT image value, with the pixels on the largest vessel at the beginning of the
    list.

    Args:
        point (tuple(int, int)): Point on centerline.
        radius (int): Radius of search circle.
        visited (np.array(bool)): Boolean array of visited image pixels.

    Returns:
        list: A list of points that can be on the vessel centerline.
    """
    # get points on circle circumference
    circle_points = circle_perimeter(*point, radius=radius, method="andres")
    # create image with edt values of point on circle circumference
    helper = np.zeros_like(edt_img)
    helper[circle_points] = edt_img[circle_points]
    # find local maxima
    coords = peak_local_max(
        helper, min_distance=radius // 3, threshold_abs=2, threshold_rel=0.5
    )
    # find coordinates of local maxima
    helper[circle_points] = 0
    helper[tuple(coords.T)] = 1
    helper[visited] = 0
    points_of_interest = list(zip(*np.nonzero(helper)))
    # sort points by vessel diameter
    points_of_interest.sort(key=lambda x: edt_img[x], reverse=True)
    return points_of_interest


def get_vessel_graph(seg_img):
    """Produce a graph which describes the vessel structure.

    Args:
        seg_img (np.array): Segmentation of vessel image.

    Returns:
        nx.Graph: Directed graph describing the vessel structure.
    """
    # perform euclidean distance transform on base image
    edt_img = distance_transform_edt(seg_img)
    # get starting points
    starting_point_stack = get_starting_points(edt_img)
    # create vessel tree data structure
    vessel_graph = nx.DiGraph()
    # create array for holding information on visited pixels
    visited = np.zeros_like(edt_img).astype(bool)

    while starting_point_stack:
        # select new starting point and check if it was not visited already
        starting_point = starting_point_stack.pop()
        if visited[starting_point]:
            continue

        vessel_graph.add_node(starting_point)
        points_to_examine = [starting_point]

        while points_to_examine:
            # get point from queue
            point = points_to_examine.pop()
            # calculate vessel radius
            radius = int(edt_img[point])
            # if radius is too small, don't go further into this vessel
            # also limited by min_distance having to be >= 1 in peak_local_max
            if radius <= 2:
                continue
            # get points that can be on centreline
            points_of_interest = get_points_of_interest(point, radius, visited, edt_img)
            # eliminate points that are too close to each other
            for poi_1 in points_of_interest:
                for poi_2 in points_of_interest:
                    if poi_1 != poi_2 and euclidean(poi_1, poi_2) < edt_img[poi_1]:
                        points_of_interest.pop(points_of_interest.index(poi_2))
            # add points of interest to examination list and to vessel graph
            for poi in points_of_interest:
                # avoid node duplication
                if not (poi in points_to_examine or vessel_graph.has_node(poi)):
                    points_to_examine.append(poi)
                    vessel_graph.add_edge(point, poi)
            # remove potential centreline pixels next to analyzed point to prevent going backwards
            disk_points = disk(point, radius)
            visited[disk_points] = True

    return vessel_graph.to_undirected()


def remove_graph_components(graph):
    """Remove small (less than 4 nodes) graph components.

    Args:
        graph (nx.Graph): Undirected graph object from NetworkX library.

    Returns:
        nx.Graph: Graph without small components.
    """
    # get components
    components = list(nx.connected_components(graph))
    # remove small components
    for component in components:
        if len(component) < 4:
            for node in component:
                graph.remove_node(node)
    return graph


# pylint: disable=too-many-boolean-expressions,invalid-name
def connect_endings(graph, edt_img, multiplier=2.5):
    """Fix discontinuities in vessel graph.

    Args:
        graph (nx.Graph): Graph object from NetworkX library.
        edt_img (np.array): Numpy array containing euclidean distance transformed image of vessel.
        multiplier (int, optional): Multiplier for maximum distance between nodes to connect.
        Defaults to 2.

    Returns:
        nx.Graph: Graph without discontinuities.
    """
    # find potential discontinuities
    endings = [node for node, degree in graph.degree() if degree == 1]
    # for every ending run BFS connection search
    while endings:
        # get point to find connection for
        start = endings.pop()
        # calculate search area
        r = edt_img[start] * multiplier
        search_area = int(r * r * np.pi)
        # setup BFS
        points = SimpleQueue()
        points.put(start)
        visited = []
        # run BFS on a restricted area
        while not points.empty() and search_area:
            search_area -= 1
            # get point
            x, y = points.get()
            visited.append((x, y))

            # check if point is a node and is a valid connection
            if graph.has_node((x, y)) and not nx.has_path(graph, start, (x, y)):
                graph.add_edge(start, (x, y))
                # this is to prevent accidentally creating bifurcations
                if (x, y) in endings:
                    endings.pop(endings.index((x, y)))
                break

            # add point to search if it is in segmentation mask and it is not visited
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    new_point = (x + dx, y + dy)
                    if (
                        x + dx >= 0
                        and x + dx < edt_img.shape[0]
                        and y + dy >= 0
                        and y + dy < edt_img.shape[1]
                        and new_point not in visited
                        and edt_img[x + dx, y + dy] > 0
                    ):
                        visited.append(new_point)
                        points.put(new_point)

    return graph


def get_node_data(node, prev_node, next_node, edt_img):
    """Get data for node in vessel graph.

    Args:
        node (tuple): Current node for which data is extracted.
        prev_node (tuple): Previous node in graph traversal.
        next_node (tuple): Next node in graph traversal.
        edt_img (np.array): Euclidean distance transformed image of segmented vessel.

    Returns:
        dict: Dictionary containing node data.
    """
    data = {}
    # get vessel diameter
    data["vessel_diameter"] = float(edt_img[node])
    # get vessel diameter gradient
    data["vessel_diameter_grad"] = float(edt_img[node] - edt_img[prev_node])
    # get vessel fragment length
    v1 = np.subtract(prev_node, node).astype(np.float32)
    data["vessel_length"] = float(np.linalg.norm(v1))
    # get angle between nodes
    v1 /= np.linalg.norm(v1)
    v2 = np.subtract(next_node, node).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    data["angle"] = float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return data


def parametrize_graph(vessel_graph, edt_img):
    """Retrieve general segment data from detailed vessel graph.

    Args:
        vessel_graph (nx.Graph): Graph describing detailed vessel structure.
        edt_img (np.array): Numpy array containing euclidean distance transformed segmentation of
        vessel.

    Returns:
        nx.Digraph: Directed, parametrized graph containing vessel segment data in nodes.
    """
    # get nodes and sort them by vessel diameter
    # this list can be used in the future to include disconnected graph components
    nodes = [node for node, degree in vessel_graph.degree() if degree == 1]
    nodes.sort(key=lambda n: edt_img[n])
    # select starting point
    start = nodes.pop()
    # initiate graph
    data_graph = nx.Graph()
    # this variable is for identifying nodes in parametrized graph
    data_graph.add_node(start)
    # initiate paths to explore from this point
    paths_to_explore = [(start, neighbor) for neighbor in vessel_graph.neighbors(start)]
    visited = [start]
    # check every path for bifurcations or endings
    while paths_to_explore:
        # get relevant data
        start, node = paths_to_explore.pop()
        prev = start
        # flag node as visited
        visited.append(node)
        # create object to gather data from vessel segment
        data = {"nodes": {start: get_node_data(start, start, node, edt_img)}}
        # measure length of vessel segment
        total_len = 0
        vessel_diameters = []
        # traverse segment while no bifurcations or endings were detected
        while vessel_graph.degree(node) == 2:
            # get next node
            neighbors = list(vessel_graph.neighbors(node))
            neighbors.pop(neighbors.index(prev))
            next_node = neighbors[0]
            # gather data on next node
            data["nodes"][node] = get_node_data(node, prev, next_node, edt_img)
            total_len += data["nodes"][node]["vessel_length"]
            vessel_diameters.append(data["nodes"][node]["vessel_diameter"])
            # go ahead
            prev = node
            node = next_node
            visited.append(node)
        data["nodes"][node] = get_node_data(node, prev, node, edt_img)
        total_len += data["nodes"][node]["vessel_length"]
        vessel_diameters.append(data["nodes"][node]["vessel_diameter"])
        # this happens after ending/bifurcation was detected
        data["segment_length"] = total_len
        data["average_vessel_diameter"] = sum(vessel_diameters) / len(vessel_diameters)
        # add node with collected data
        data_graph.add_node(node)
        data_graph.add_edge(start, node, **data)
        # add paths to explore if there are new ones
        paths_to_explore.extend(
            [
                (node, neighbor)
                for neighbor in vessel_graph.neighbors(node)
                if neighbor not in visited
            ]
        )
    return data_graph


def choose_root_node(data_graph):
    """Select root node for data graph.

    Args:
        data_graph (nx.Graph): Undirected vessel data graph.

    Returns:
        tuple: coordinates of root node.
    """
    # select edge with biggest average vessel diameter
    edges = [
        k
        for k, _ in sorted(
            nx.get_edge_attributes(data_graph, "average_vessel_diameter").items(),
            key=lambda item: item[1],
        )
    ]
    n1, n2 = edges[-1]
    # select root by node degree
    if data_graph.degree(n1) == 1 and data_graph.degree(n2) != 1:
        return n1
    if data_graph.degree(n2) == 1 and data_graph.degree(n1) != 1:
        return n2
    # select root by biggest vessel diameter
    n1_diameter = data_graph[n1][n2]["nodes"][n1]["vessel_diameter"]
    n2_diameter = data_graph[n1][n2]["nodes"][n2]["vessel_diameter"]
    return n1 if n1_diameter > n2_diameter else n2


def clean_data_graph(data_graph, min_segment_length=20):
    """Clean data graph by removing small segments, nodes of degree 2 and selecting a new root.

    Args:
        data_graph (nx.Graph): Undirected vessel data graph.
        min_segment_length (int, optional): Minimum segment length permissible in graph.
        Defaults to 20[px].

    Returns:
        nx.Graph: Cleaned vessel data graph.
    """
    # remove small segments
    for edge, segment_length in nx.get_edge_attributes(
        data_graph, "segment_length"
    ).items():
        if segment_length < min_segment_length:
            data_graph.remove_edge(*edge)
    # remove isolated nodes (degree == 0)
    data_graph.remove_nodes_from(list(nx.isolates(data_graph)))
    # merge segments for nodes of degree 2
    while [n for n, degree in data_graph.degree() if degree == 2]:
        for node, degree in list(data_graph.degree()):
            if degree != 2:
                continue
            n1, n2 = list(data_graph.neighbors(node))
            new_edge_data = {}
            new_edge_data["nodes"] = (
                data_graph[node][n1]["nodes"] | data_graph[node][n2]["nodes"]
            )
            new_edge_data["segment_length"] = sum(
                node_data["vessel_length"]
                for node_data in new_edge_data["nodes"].values()
            )
            new_edge_data["average_vessel_diameter"] = sum(
                node_data["vessel_diameter"]
                for node_data in new_edge_data["nodes"].values()
            ) / len(new_edge_data["nodes"])
            data_graph.remove_node(node)
            data_graph.add_edge(n1, n2, **new_edge_data)
    # select new root
    root_node = choose_root_node(data_graph)
    nx.set_node_attributes(data_graph, False, "root")
    data_graph.nodes[root_node]["root"] = True
    return data_graph


def identify_segments(data_graph, primary_angle, secondary_angle):
    """Identify vessel segments in vessel data graph.

    Args:
        data_graph (nx.Graph): Undirected vessel data graph.
        primary_angle (int): Positioner primary angle.
        secondary_angle (int): Positioner secondary angle.

    Raises:
        AttributeError: if angles are outside of the expected bounds (at least 20 degrees off the
        expected values defined in PROJECTION_ANGLES).

    Returns:
        nx.Graph: vessel data graph with labeled vessel segments.
    """
    for angles, vessel_labels in PROJECTION_ANGLES.items():
        primary, secondary = angles
        if (
            abs(primary_angle - primary) <= 20
            and abs(secondary_angle - secondary) <= 20
        ):
            labels = vessel_labels.copy()
            break
    else:
        raise AttributeError("Image has incorrect projection angles.")

    for k, v in data_graph.nodes(data="root"):
        if v:
            root_node = k
            break

    current_label = ""
    for n1, n2 in nx.bfs_edges(data_graph, root_node):
        if labels:
            current_label = labels.pop(0)
        data_graph[n1][n2]["vessel_label"] = current_label

    return data_graph


def get_segment_mask(data_graph, seg_img):
    """Generate representation of vessel segments in the form of segmentation masks.

    Args:
        data_graph (nx.Graph): NetworkX graph containing general vessel information.
        seg_img (np.array): Numpy array containing segmented vessel image.

    Returns:
        tuple(np.array, dict): Numpy array containing segment mask and label dict.
    """
    masks = []
    labels = {}
    for idx, nodes in enumerate(data_graph.edges.data("nodes")):
        try:
            labels[idx + 1] = data_graph[nodes[0]][nodes[1]]["vessel_label"]
        except KeyError:
            labels[idx + 1] = "UNKN"
            # WARN print(idx,nodes[0],nodes[1]," no label")
        # calculate distances from nodes in segment
        mask = np.ones_like(seg_img)
        mask[nodes[:-1]] = 0
        idxs = tuple(np.array(list(nodes[-1].keys())).T)
        mask[idxs] = 0
        mask = distance_transform_edt(mask)
        # append to summary mask
        masks.append(mask)
    masks = np.stack(masks)
    # select label closest to point
    result = np.argmin(masks, axis=0) + 1
    result[seg_img == 0] = 0
    result = result.astype(np.uint8)
    return result, labels


def get_data_graph(seg_img):
    """Function collecting steps for creating a data graph.

    Args:
        seg_img (np.array): Numpy array containing segmented vessel image.

    Returns:
        nx.Graph: Vessel data graph.
    """
    edt_img = distance_transform_edt(seg_img)
    vessel_graph = get_vessel_graph(seg_img)
    vessel_graph = remove_graph_components(vessel_graph)
    vessel_graph = connect_endings(vessel_graph, edt_img)
    data_graph = parametrize_graph(vessel_graph, edt_img)
    data_graph = clean_data_graph(data_graph)
    return data_graph
