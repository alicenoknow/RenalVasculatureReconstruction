import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
from skimage import measure, segmentation, feature
from vis_utils_no_print import load_volume, VolumeVisualizer, ColorMapVisualizer
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
from skimage.morphology import skeletonize_3d, binary_dilation, convex_hull_image
from skimage import filters, morphology
from scipy import signal
from skimage.filters import frangi, sato
from PIL import Image
from tqdm import tqdm
import os
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import confusion_matrix

from scipy.ndimage import zoom
import re

s_kernel_sizes = {
    'P01': range(0, 14)
}

s_number_of_iterations = {
    'P01': 7
}

thresholds = {
    'P01': 21,
    'P02': 28,
    'P03': 160,
    'P04': 30,
    'P05': 24,
    'P06': 22,
    'P07': 100,
    'P08': 38,#"Not usable",
    'P09': 70,
    'P10': 50,
    'P11': 50,
    'P12': 74,
    'P13': 45,
    'P14': 42,
    'P15': 95,
    'P16': 25,
    'P17': 80,
    'P18': 85,
    'P19': 50,
    'P20': 95,
    'P21': 45,
    'P22': 40,#"Weird artefact",
    'P23': 60,
    'P24': 80,
    'P25': 70,
    'P26': 130,
    'P27': 70,
    'P28': 25,
    
    'P29': 120, #TODO for now seems unusable, too clutered
    'P30': 80,
    'P31': 50,
    'P32': 48,
    'P33': 65,
}

thresholds_m = {
    'M01': 30,
    'M09': 25,
    'M18': 35, #??
}

def get_volume(filename):
    try:
        tokens = re.split(r'x|_|\.', filename)
        shape_z, shape_y, shape_x = int(tokens[-4]), int(tokens[-3]), int(tokens[-2])
        volume = np.fromfile(filename, dtype=np.uint8)
        return volume.reshape(shape_x, shape_y, shape_z)
    except:
        print("Invalid filename, correct format: <filename>_<shape x>x<shape y>x<shape z>.raw")


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
        kernel_sizes_maps.append(kernel_size_map)

    return kernel_sizes_maps


def reconstruct_with_finetuning(filename, tree_name):
    
    if filename[-3:] == 'raw':
        volume = load_volume(filename='../data/' + tree_name + '/' + filename)
    else:
        volume = np.load(filename)
    
    if tree_name[0] == 'M':
        mask = volume > thresholds_m.get(tree_name, 25)
    else:
        mask = volume > thresholds.get(tree_name, 25)

    main_region_min_size = {
    'P01': 30_000,
    'P03': 50_000,
    'P07': 300_000,
    'P17': 300_000,
    }

    main_regions, bounding_boxes = get_main_regions(mask, min_size=main_region_min_size.get(tree_name, 20_000))
    mask_main = main_regions
    s_recos = calculate_reconstruction(mask_main, 
                                   kernel_sizes=s_kernel_sizes.get(tree_name, range(0, 13)), 
                                   iters=s_number_of_iterations.get(tree_name, 3))
    s_reco = s_recos[-1] > 0
    pth = './reconstructed/PREP_'
    if tree_name[0] == 'M':
        pth = './reconstructed/MODEL_'
    np.save(pth + tree_name, np.array(s_reco))
    
    shape = np.min([mask_main.shape, s_reco.shape], axis=0)

    mask_main_r = mask_main[:shape[0], :shape[1], :shape[2]]
    u_reco_r = s_reco[:shape[0], :shape[1], :shape[2]]
    joint_reco = np.logical_or(mask_main_r, u_reco_r)

    main_regions, bounding_boxes = get_main_regions(joint_reco, min_size=main_region_min_size.get(tree_name, 20_000))
    new_joint_reco = np.logical_or(mask_main_r, main_regions)
    
    pth = './reconstructed_P_full/PREP_'
    if tree_name[0] == 'M':
        pth = './reconstructed_M_full/MODEL_'
    np.save(pth + tree_name, np.array(new_joint_reco))
    return volume, s_reco


if __name__ == '__main__':
    dir_path = '../data/P*'
    files = glob.glob(dir_path + '/P*.raw')
    files = sorted(files)[11:]
    print(files)
    report = ''
    for file in files:
        tree_name =  file.split('\\')[-2][:3]
        print(tree_name)
        print('##################')
        original, recons = reconstruct_with_finetuning(file, tree_name)
