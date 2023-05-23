import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
from skimage import measure, segmentation, feature
from vis_utils import load_volume, VolumeVisualizer, ColorMapVisualizer
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
from skimage.morphology import skeletonize_3d, binary_dilation, convex_hull_image
from skimage import filters, morphology
from scipy import signal
from skimage.filters import frangi, sato
from PIL import Image
from tqdm import tqdm
from stl import mesh
import os
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

from scipy.ndimage import zoom

s_kernel_sizes = {
    'P01': range(0, 14)
}

s_number_of_iterations = {
    'P01': 7
}

def load_volume_from_mesh(filename):
    mesh_ = mesh.Mesh.from_file(filename)
    x_min, y_min, z_min = mesh_.min_
    x_max, y_max, z_max = mesh_.max_
    x_dim, y_dim, z_dim = int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)

    volume = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint8)

    for triangle in mesh_.vectors:
        x, y, z = np.transpose(triangle)
        x -= x_min
        y -= y_min
        z -= z_min
        indices = np.vstack((x, y, z)).T.astype(int)
        volume[tuple(indices[:, [0, 1, 2]].T)] = 1
    return volume

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

def calculate_reconstruction(mask, kernel_sizes=[10, 9, 8], fill_threshold=0.5, iters=1, conv_dtype=np.uint16, fft=True):
    kernel_sizes_maps = []
    mask = mask.astype(np.uint8)
    
    for i in range(iters):
        kernel_size_map = np.zeros(mask.shape, dtype=np.uint8)

        for kernel_size in kernel_sizes:
            fill_percentage = convolve_with_ball(mask, kernel_size, dtype=conv_dtype, normalize=True, fft=fft)
            
            above_threshold_fill_indices = fill_percentage > fill_threshold
            kernel_size_map[above_threshold_fill_indices] = kernel_size + 1

            mask[above_threshold_fill_indices] = 1
            
            print(f'Iteration {i + 1} kernel {kernel_size} done')

        kernel_sizes_maps.append(kernel_size_map)
        print(f'Iteration {i + 1} ended successfully')

    return kernel_sizes_maps

def dice_coefficient(img1, img2):
    intersection = np.logical_and(img1, img2)
    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def reconstruct(filename, tree_name):
    
    volume = load_volume_from_mesh(filename='../data/models/' + filename)
    mask = volume
    main_region_min_size = {
    'MODEL_15': 12_800
    }

    main_regions, bounding_boxes = get_main_regions(mask, min_size=main_region_min_size.get(tree_name, 1_000))
    mask_main = main_regions
    s_recos = calculate_reconstruction(mask_main, 
                                   kernel_sizes=s_kernel_sizes.get(tree_name, range(0, 13)), 
                                   iters=s_number_of_iterations.get(tree_name, 3))
    s_reco = s_recos[-1] > 0
    np.save('../data/reconstructed/' + tree_name, np.array(s_reco))
    return volume, s_reco


def evaluate(original_filename, reconstruction_filename, tree):
    reconstuction = np.load('../data/reconstructed/' + reconstruction_filename, allow_pickle=True)
    original = load_volume_from_mesh('../data/models/' + original_filename)
    report = tree + '\n\n'
    dice_score = dice_coefficient(original, reconstuction)
    report += "Dice coefficient: " +  str(dice_score) + '\n\n'

    original_points_list = np.array(tuple(zip(*np.nonzero(original))))
    reconstr_points_list = np.array(tuple(zip(*np.nonzero(reconstuction))))
    hausdorff1 = directed_hausdorff(original_points_list, reconstr_points_list)
    hausdorff2 = directed_hausdorff(reconstr_points_list, original_points_list)
    report += 'Hausdorff distance: ' + str(max(hausdorff1, hausdorff2)) + '\n\n'

    original_flat = np.ravel(original)
    recons_flat = np.ravel(reconstuction)

    conf_mat = confusion_matrix(original_flat, recons_flat)

    TN, FP, FN, TP = conf_mat.ravel()
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    report += 'Precission: ' + str(precision) + '\nAccuracy: ' + str(accuracy) + '\n'

    report += 'Confussion matrix: \n' + str(conf_mat) + '\n\n ################################### \n\n'

    return report


if __name__ == '__main__':
    files = os.listdir('../data/models')
    files = sorted(files)
    report = ''
    for file in files:
        tree_name = 'MODEL_' + file[:2]
        # original, recons = reconstruct(file, tree_name)
        # report += evaluate2(original=original, reconstuction=recons, tree=tree_name)
        report += evaluate(original_filename=file, reconstruction_filename=tree_name + '.npy', tree=tree_name)
    with open("report.txt", "w") as rep_file:
        rep_file.write(report)


    
