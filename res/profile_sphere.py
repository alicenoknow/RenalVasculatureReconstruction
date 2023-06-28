from vis_utils import load_volume, VolumeVisualizer, ColorMapVisualizer
from identification import get_vessel_graph, remove_graph_components, parametrize_graph, clean_data_graph
import numpy as np
import networkx as nx
import pickle

from queue import SimpleQueue
from skimage import measure, morphology 
from scipy.ndimage import distance_transform_edt, zoom
from scipy.signal import fftconvolve
from scipy.spatial.distance import euclidean
import cProfile
import re

def get_volume(filename):
    try:
        tokens = re.split(r'x|_|\.', filename)
        shape_z, shape_y, shape_x = int(tokens[-4]), int(tokens[-3]), int(tokens[-2])
        volume = np.fromfile(filename, dtype=np.uint8)
        return volume.reshape(shape_x, shape_y, shape_z)
    except:
        print("Invalid filename, correct format: <filename>_<shape x>x<shape y>x<shape z>.raw")
        
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
            
#             print(f'Iteration {i + 1} kernel {kernel_size} done')

        kernel_sizes_maps.append(kernel_size_map)
#     print("reco end")

    return kernel_sizes_maps

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

def connect_endings_3d(graph, edt_img, multiplier=2.5):
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
    i = 0
#     print("Endings: ", len(endings))
    while endings:
        # get point to find connection for
        start = endings.pop()
        # calculate search area
        r = edt_img[start] * multiplier
        search_area = int(4/3 * r * r * r * np.pi)
        # setup BFS
        points = SimpleQueue()
        points.put(start)
        visited = np.zeros_like(edt_img).astype(bool)
        
        # run BFS on a restricted area
        while not points.empty() and search_area:
            search_area -= 1
            # get point
            x, y, z = points.get()
            visited[x, y, z] = True
            
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
                            and not visited[new_point[0], new_point[1], new_point[2]]
                            and edt_img[x + dx, y + dy, z + dz] > 0
                        ):
                            visited[new_point[0], new_point[1], new_point[2]] = True
                            points.put(new_point)

    return graph

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
            new_edge_data["nodes"] = data_graph[node][n1]["nodes"]
            new_edge_data["nodes"].update(data_graph[node][n2]["nodes"])
            
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


def main():
    volume = get_volume("../data/P13/P13_60um_1132x488x877.raw")
    mask = volume > 32

    main_regions, bounding_boxes = get_main_regions(mask, min_size=25_000)

    scaled_mask = zoom(main_regions, zoom=0.7, order=0)

    s_recos = calculate_reconstruction(scaled_mask, 
                                   kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                                   iters=3)

    bin_reco = s_recos[-1] > 0
    edt_img = distance_transform_edt(bin_reco)
    vessel_graph = get_vessel_graph(bin_reco, 3)

    vessel_graph_rm = remove_graph_components(vessel_graph)


    vessel_graph_cn = connect_endings_3d(vessel_graph, edt_img, 3)
    data_graph = parametrize_graph(vessel_graph_cn, edt_img)
    data_graph_cl = clean_data_graph(data_graph)
    
    pickle.dump(data_graph_cl, open('profile.pickle', 'wb'))

    

profiler = cProfile.Profile()
profiler.enable()
  
main()

profiler.disable()
profiler.dump_stats("sphere_opt_full.prof")


