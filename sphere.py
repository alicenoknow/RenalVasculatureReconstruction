import pickle
import numpy as np
import networkx as nx

from skimage import measure, segmentation, feature, filters, morphology
from skimage.morphology import skeletonize_3d, binary_dilation, convex_hull_image, ball
from skimage.draw import circle_perimeter, disk
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, zoom
from scipy.signal import fftconvolve
from scipy.spatial.distance import euclidean

from queue import SimpleQueue

def get_main_regions(binary_mask, min_size=10_000, connectivity=3):
    labeled = measure.label(binary_mask, connectivity=connectivity)
    region_props = measure.regionprops(labeled)
    
    main_regions = np.zeros(binary_mask.shape)
    bounding_boxes = []
    for props in region_props:
        if props.area >= min_size:
            bounding_boxes.append(props.bbox)
            main_regions = np.logical_or(main_regions, labeled==props.label)
            
    lower_bounds = np.min(bounding_boxes, axis=0)[:3]
    upper_bounds = np.max(bounding_boxes, axis=0)[3:]

    return main_regions[
        lower_bounds[0]:upper_bounds[0],
        lower_bounds[1]:upper_bounds[1],
        lower_bounds[2]:upper_bounds[2],
    ], bounding_boxes

def spherical_kernel(outer_radius, thickness=1, filled=True):    
    outer_sphere = morphology.ball(radius=outer_radius)
    if filled:
        return outer_sphere
    
    thickness = min(thickness, outer_radius)
    
    inner_radius = outer_radius - thickness
    inner_sphere = morphology.ball(radius=inner_radius)
    
    begin = outer_radius - inner_radius
    end = begin + inner_sphere.shape[0]
    outer_sphere[begin:end, begin:end, begin:end] -= inner_sphere
    return outer_sphere

def convolve_with_ball(img, ball_radius, dtype=np.uint16, normalize=True, fft=True):
    kernel = spherical_kernel(ball_radius, filled=True)
    if fft:
        convolved = fftconvolve(img.astype(dtype), kernel.astype(dtype), mode='same')
    else:
        convolved = signal.convolve(img.astype(dtype), kernel.astype(dtype), mode='same')
    
    if not normalize:
        return convolved
    
    return (convolved / kernel.sum()).astype(np.float16)

def calculate_reconstruction(mask, kernel_sizes=[10, 9, 8, 7], fill_threshold=0.5, iters=1, conv_dtype=np.uint16, fft=True):
    kernel_sizes_maps = []
    mask = mask.astype(np.uint8)
    
    for i in range(iters):
        kernel_size_map = np.zeros(mask.shape, dtype=np.uint8)

        for kernel_size in kernel_sizes:
            fill_percentage = convolve_with_ball(mask, kernel_size, dtype=conv_dtype, normalize=True, fft=fft)
            
            above_threshold_fill_indices = fill_percentage > fill_threshold
            kernel_size_map[above_threshold_fill_indices] = kernel_size + 1

            mask[above_threshold_fill_indices] = 1
        kernel_sizes_maps.append(kernel_size_map)
    return kernel_sizes_maps


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


def get_sphere_points(center, radius, im_size):
    size = np.min([np.max(center) + 2*radius -1, im_size[0]-1, im_size[1]-1, im_size[2]-1])
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    d = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
    surface_points = np.argwhere(np.isclose(d, radius))
    return [tuple(point) for point in surface_points]

def sphere_perimeter(coords, radius, shape):
    # Create arrays of indices for each dimension
    x_indices = np.arange(shape[0])
    y_indices = np.arange(shape[1])
    z_indices = np.arange(shape[2])
    
    # Create a 3D grid of indices for each dimension
    x_grid, y_grid, z_grid = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    
    # Calculate the distance of each point from the center
    distances = np.sqrt((x_grid - coords[0])**2 + (y_grid - coords[1])**2 + (z_grid - coords[2])**2)
    
    # Find the indices of points that are on the sphere surface
    sphere_indices = np.where(np.isclose(distances, radius))
    
    return sphere_indices

def sphere_full(coords, radius, shape):
    # Create arrays of indices for each dimension
    x_indices = np.arange(shape[0]-1)
    y_indices = np.arange(shape[1]-1)
    z_indices = np.arange(shape[2]-1)
    
    # Create a 3D grid of indices for each dimension
    x_grid, y_grid, z_grid = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    
    # Calculate the distance of each point from the center
    distances = np.sqrt((x_grid - coords[0])**2 + (y_grid - coords[1])**2 + (z_grid - coords[2])**2)
    
    sphere_indices = np.where(distances < radius)
    
    # Create a binary mask with True values at the indices of points on the sphere surface
    mask = np.zeros(shape, dtype=bool)
    mask[sphere_indices] = True
    
    return mask

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
    #circle_points = circle_perimeter(*point, radius=radius, method="andres")
    circle_points = sphere_perimeter(point, radius, edt_img.shape)
    #print("Circle points: ", len(circle_points), edt_img.shape)
    # create image with edt values of point on circle circumference
    helper = np.zeros_like(edt_img)
    helper[circle_points] = edt_img[circle_points]
    # find local maxima
    coords = peak_local_max(
        helper, min_distance=radius // 3, threshold_abs=2, threshold_rel=0.5
    )
    #print("Local peaks on surface: ", len(coords))
    # find coordinates of local maxima
    helper[circle_points] = 0
    helper[tuple(coords.T)] = 1
    helper[visited] = 0
    points_of_interest = list(zip(*np.nonzero(helper)))
    # sort points by vessel diameter
    points_of_interest.sort(key=lambda x: edt_img[x], reverse=True)
    return points_of_interest


def get_disk_3d(center, radius):
    size = np.max(center) + 2*radius
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    d = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
    surface_points = np.argwhere(d <= radius)
    return [tuple(point) for point in surface_points]



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
            disk_points = sphere_full(point, radius // 2, visited.shape)

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
def connect_endings(graph, edt_img, multiplier=2.):
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

# pylint: disable=too-many-boolean-expressions,invalid-name
def connect_endings_3d(graph, edt_img, multiplier=1):
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
    endings = [node for node, degree in graph.degree() if degree <= 1]
    # for every ending run BFS connection search
    while endings:
        # get point to find connection for
        start = endings.pop()
        # calculate search area
        r = edt_img[start] * multiplier
        search_area = int(4/3 * r * r * r * np.pi)
        # setup BFS
        points = SimpleQueue()
        points.put(start)
        visited = []
        # run BFS on a restricted area
        while not points.empty() and search_area:
            search_area -= 1
            # get point
            x, y, z = points.get()
            visited.append((x, y, z))

            # check if point is a node and is a valid connection
            if graph.has_node((x, y, z)) and not nx.has_path(graph, start, (x, y, z)):
                graph.add_edge(start, (x, y, z))
                # this is to prevent accidentally creating bifurcations
                if (x, y, z) in endings:
                    endings.pop(endings.index((x, y, z)))
                break

            # add point to search if it is in segmentation mask and it is not visited
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        new_point = (x + dx, y + dy, z + dz)
                        if (
                            x + dx >= 0
                            and x + dx < edt_img.shape[0]
                            and y + dy >= 0
                            and y + dy < edt_img.shape[1]
                            and z + dz >= 0
                            and z + dz < edt_img.shape[2]
                            and new_point not in visited
                            and edt_img[x + dx, y + dy, z + dz] > 0
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


volume = np.fromfile('../data/P13/data.raw', dtype=np.uint8)
volume = volume.reshape(877, 488, 1132)
mask = volume > 32

main_regions, bounding_boxes = get_main_regions(mask, min_size=25_000)
scaled_mask = zoom(main_regions, zoom=0.5, order=1)

s_recos = calculate_reconstruction(scaled_mask, 
                                   kernel_sizes=[3, 4, 5, 6, 7, 8, 9], 
                                   iters=5)

bin_reco = s_recos[-1] > 0

edt_img = distance_transform_edt(bin_reco)
vessel_graph = get_vessel_graph(bin_reco)

pickle.dump(vessel_graph, open('vessel_graph_script.pickle', 'wb'))

vessel_graph_cn = connect_endings_3d(vessel_graph, edt_img)

pickle.dump(vessel_graph, open('vessel_graph_connected_script.pickle', 'wb'))

