#!/usr/bin/env python
# coding: utf-8


import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
from skimage import measure, segmentation, feature
from vis_utils import load_volume, VolumeVisualizer, ColorMapVisualizer
from scipy.ndimage import zoom
from scipy.signal import fftconvolve, oaconvolve
from skimage.morphology import skeletonize_3d, binary_dilation, convex_hull_image
from skimage import filters, morphology
from scipy import signal
from skimage.filters import frangi, sato
from PIL import Image
from tqdm import tqdm

from scipy.ndimage import zoom

import cProfile


def visualize_addition(base, base_with_addition):
    base = (base.copy() > 0).astype(np.uint8)
    addition = (base_with_addition > 0).astype(np.uint8)
    addition[base == 1] = 0
    ColorMapVisualizer(base + addition * 4).visualize()
    
def visualize_lsd(lsd_mask):
    ColorMapVisualizer(lsd_mask.astype(np.uint8)).visualize()
    
def visualize_gradient(lsd_mask):
    ColorMapVisualizer(lsd_mask.astype(np.uint8)).visualize(gradient=True)
    
def visualize_mask_bin(mask):
    VolumeVisualizer((mask > 0).astype(np.uint8), binary=True).visualize()
    
def visualize_mask_non_bin(mask):
    VolumeVisualizer((mask > 0).astype(np.uint8) * 255, binary=False).visualize()
    
def visualize_skeleton(mask, visualize_mask=True, visualize_both_versions=False):
    skeleton = skeletonize_3d((mask > 0).astype(np.uint8))
    if not visualize_mask or visualize_both_versions:
        VolumeVisualizer(skeleton, binary=True).visualize()
    if visualize_mask or visualize_both_versions:
        skeleton = skeleton.astype(np.uint8) * 4
        mask = (mask > 0).astype(np.uint8) * 3
        mask[skeleton != 0] = 0
        ColorMapVisualizer(skeleton + mask).visualize()

def visualize_ultimate(lsd, base_mask):
    visualize_lsd(lsd)
    visualize_mask_non_bin(lsd)
    visualize_addition(base_mask, lsd)
    visualize_skeleton(lsd, visualize_mask=True)


def f(mask, scale, order=0):
    return zoom(mask, scale, order=order)
    
def verify_mask(mask):
    regions_count = np.max(measure.label(mask, connectivity=3))
    print("one-piece mask" if regions_count == 1 else f"scattered mask, number of regions: {regions_count}")


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
        print(f'Iteration {i + 1} ended successfully')

    return kernel_sizes_maps


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



TREE_NAME = 'P13'

profiler = cProfile.Profile()
profiler.enable()

volume = np.fromfile('../data/P13/P13_60um_1132x488x877.raw', dtype=np.uint8)
volume = volume.reshape(877, 488, 1132)

mask = volume > thresholds[TREE_NAME]

main_region_min_size = {
    'P01': 30_000,
    'P03': 50_000,
    'P07': 300_000,
    'P17': 300_000,
}


main_regions, bounding_boxes = get_main_regions(mask, min_size=main_region_min_size.get(TREE_NAME, 25_000))
mask_main = main_regions


mask = zoom(mask_main, 0.7, order=0)


s_kernel_sizes = {
    'P01': range(0, 14),
    'P02': range(0, 14),
#     'P03': range(0, 13),
#     'P04': range(0, 13),
#     'P05': range(0, 13),
#     'P06': range(0, 13),
    'P07': range(0, 14),
    'P09': range(0, 12),
    'P10': range(0, 12),
#     'P11': range(0, 13),
#     'P12': range(0, 13),
    'P13': range(0, 16),
#     'P14': range(0, 13),
    'P15': range(0, 14),
#     'P16': range(0, 13),
    'P17': range(0, 14),
#     'P18': range(0, 13),
    'P19': range(0, 12),
#     'P20': range(0, 13),
    'P21': range(0, 12),
    'P23': range(0, 14),
#     'P24': range(0, 13),
#     'P25': range(0, 13),
#     'P26': range(0, 13),
#     'P27': range(0, 13)
    'P28': range(0, 14),
}

s_number_of_iterations = {
    'P01': 7,
    'P02': 1,
    'P03': 10,
    'P04': 2,
    'P05': 3,
    'P06': 5,
    'P07': 4,
    'P09': 6,
    'P10': 5,
    'P11': 4, 
    'P12': 3,
    'P13': 3,
    'P14': 2,
    'P15': 3,
    'P16': 5,
    'P17': 10,
    'P18': 4,
    'P19': 2,
    'P20': 1,
    'P21': 3,
    'P23': 5,
    'P24': 2,
    'P25': 6,
    'P26': 6,
    'P27': 1,
    'P28': 1,
    'P30': 1,
    'P31': 3,
    'P32': 3,
    'P33': 3,
}

s_recos = calculate_reconstruction(mask, 
                                   kernel_sizes=s_kernel_sizes.get(TREE_NAME, range(0, 13)), 
                                   iters=s_number_of_iterations.get(TREE_NAME, 3))


s_reco = s_recos[-1] > 0

def upscale_reconstruction(s_reco, scale=0.7):
    upscale = 1 / scale
    u_reco = zoom(s_reco, upscale, order=3)
    return u_reco

u_reco = upscale_reconstruction(s_reco, scale=0.4)

shape = np.min([mask_main.shape, u_reco.shape], axis=0)

mask_main_r = mask_main[:shape[0], :shape[1], :shape[2]]
u_reco_r = u_reco[:shape[0], :shape[1], :shape[2]]
joint_reco = np.logical_or(mask_main_r, u_reco_r)

main_regions, bounding_boxes = get_main_regions(joint_reco, min_size=main_region_min_size.get(TREE_NAME, 20_000))
new_joint_reco = np.logical_or(mask_main_r, main_regions)


profiler.disable()
profiler.dump_stats("conv_full.prof")

visualize_addition(mask_main_r, new_joint_reco)