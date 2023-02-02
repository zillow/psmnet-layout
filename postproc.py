import numpy as np
import cv2
import sys
import os
import shutil
import json
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import pickle as pk
from math import sin, cos, radians
import shapely.geometry as sg
import copy
import math

import logging
import math

import cv2
import numpy as np


EPS_RAD = np.deg2rad(1)


def mostly_manhattan_shape(xy_cor, image_size=[512, 1024], pixel_center=0.5):
    '''
    Run polygon postprocessing.
    Args: 
        xy_cor: set of walls. 
        image_size: 512, 1024
        pixel_center: 0.5
    Return: 
        Corner coordinates represented with XY
    '''

    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        current_cor = xy_cor[j]
        next_cor = xy_cor[next_j]

        # Case: both current and next walls non-manhttan
        if current_cor['non_manhattan'] and next_cor['non_manhattan']:
            current_cor_nm = current_cor['non_manhattan']
            current_m, current_b = current_cor_nm['m'], current_cor_nm['b']
            next_cor_nm = next_cor['non_manhattan']
            next_m, next_b = next_cor_nm['m'], next_cor_nm['b']
            if current_m is None:
                x = current_cor['val']
                y = next_m*x + next_b
            elif next_m is None:
                x = next_cor['val']
                y = current_m*x + current_b
            else:
                x = (next_b - current_b)/(current_m - next_m)
                y = current_m*x + current_b
                
        # Case: only current wall is non-manhattan
        elif current_cor['non_manhattan']:
            current_cor_nm = current_cor['non_manhattan']
            current_m, current_b = current_cor_nm['m'], current_cor_nm['b']
            if current_m is None:
                x = current_cor['val']
                y = next_cor['val']
            elif next_cor['type'] == 0:
                x = next_cor['val']
                y = current_m*x + current_b

            else:
                next_m = 0
                next_b = next_cor['val']
                x = (next_b - current_b)/(current_m - next_m)
                #y = current_m*x + current_b
                y = next_b
                

        elif next_cor['non_manhattan']:
            next_cor_nm = next_cor['non_manhattan']
            next_m, next_b = next_cor_nm['m'], next_cor_nm['b']
            if next_m is None:
                x = next_cor['val']
                y = current_cor['val']
            elif current_cor['type'] == 0:
                x = current_cor['val']
                y = next_m*x + next_b

            else:
                y = current_cor['val']
                current_m = 0
                current_b = y
                x = (next_b - current_b)/(current_m - next_m)                

        elif current_cor['type'] == 0:
            x = current_cor['val']
            y = next_cor['val']
        else:
            y = current_cor['val']
            x = next_cor['val']

        cor.append((x, y))
    return cor


def detect_curvature(poly, length_threshold=.01, angle_threshold=30):
    '''
    Try to detect curvature in input to avoid manhattanizing these 
    sections of room shape.
    Args: 
        poly: Room shape polygon coordinates in xy. 
        length_threshold: Minimum wall length for curvature
        angle_threshold: Maximum corner angle for curvature
    Return: 
        Tuple of:
            Presence of curvature, true or false
            Mask where curvature occurs in polygon (if present)
    '''    

    CONTIGUOUS_THRESHOLD = 4   # Minimum number of points to be considered curvature 
    angles = []
    lengths = []
    contiguous_curve = 0
    max_contiguous = 0
    curvature_mask = []
    curve_mask = []
    #Loop over polygon corners
    for vi in range(2*poly.shape[0]):
        # Get set of corner coordinates
        pt_indices = [
            (vi - 2) % len(poly),
            (vi - 1) % len(poly),
            (vi) % len(poly)
        ]
        p1, p2, p3 = [poly[idx] for idx in pt_indices]        

        # Get wall segment vectors
        v1 = np.array(p2) - np.array(p1)
        length1 = np.linalg.norm(v1)
        v2 = np.array(p3) - np.array(p2)
        length2 = np.linalg.norm(v2)        
        lengths.append(length1)
        angle = np.rad2deg(angle_between_vectors(v1, v2))
        angles.append(angle)

        # Check if angle between wall segments is within range for consideration
        # as curvature
        if length1 > length_threshold and length2 > length_threshold \
           and angle < angle_threshold and angle > 5:
            contiguous_curve += 1
            curve_mask.extend(pt_indices)
        else:
            if np.unique(curve_mask).shape[0] > CONTIGUOUS_THRESHOLD:
                curvature_mask.extend(curve_mask)
            curve_mask = []
            max_contiguous = max(contiguous_curve, max_contiguous)
            contiguous_curve = 0

    return (max_contiguous >= CONTIGUOUS_THRESHOLD, np.unique(curvature_mask).tolist())


def manhattanize_shape(room_shape, length_threshold,
                       manhattan_threshold=10,
                       plot_path=None):

    '''
    Selectively manhattanize input room shape polygon.
    Args: 
        room_shape: Room shape polygon coordinates in xy.
        manhattan_threshold: How close a corner must be to a multiple of 90 to be 
                             considered manhattan.
        plot_path: Path to save plot.
    Return: 
        Selectively manhattanized room shape polygon coordinates in xy.
    '''

    # Center on centroid
    centroid = np.array(Polygon(room_shape).centroid.coords).ravel() 
    poly = room_shape - centroid

    # Check for curvature in walls, to avoid manhattanization in these regions
    has_curvature, curvature_mask = detect_curvature(poly, length_threshold = length_threshold)
    if plot_path is not None:
        plt.plot([val[0] for val in poly], [val[1] for val in poly])
        plt.savefig(plot_path + '_manhattanized.jpeg')
        plt.clf()
        plt.close()

    # Identify wall types for mostly-manhattan postprocessing
    # Rare cases will result in two manhattan walls of the same orientation
    # in a  row, we will skip these cases (often from curvature)
    invalid_manhattan = False  
    vi = 0
    walls = []
    while vi < len(poly):
        # Get neighboring points
        pt_indices = [
            (vi - 1) % len(poly),
            (vi) % len(poly)
        ]
        p1, p2 = [poly[idx] for idx in pt_indices]

        # Create wall segment vector and check orientation
        pv1 = np.array(p2) - np.array(p1)
        wall_angle = np.rad2deg(angle_between_vectors(pv1,  np.array([1, 0]))) % 90
        wall_deviation_90 = min(wall_angle, 90 - wall_angle)
        corner_angle1 = angle_between_vectors2(p1, np.array([1, 0])) - np.pi/2
        corner_angle2 = angle_between_vectors2(p2, np.array([1, 0])) - np.pi/2
        if wall_deviation_90 < manhattan_threshold and \
           np.abs(np.dot(pv1/np.linalg.norm(pv1), np.array([1, 0]))) < \
           np.abs(np.dot(pv1/np.linalg.norm(pv1), np.array([0, 1]))):
            walls.append({'type': 0,
                          'val': np.mean([p1[0], p2[0]]),
                          'u0': corner_angle1,
                          'u1': corner_angle2,
                          'non_manhattan': None})
            if len(walls) >= 2 and walls[-2]['type'] == walls[-1]['type']:
                invalid_manhattan = True
                break
        elif wall_deviation_90 < manhattan_threshold:
            walls.append({'type': 1,
                          'val': np.mean([p1[1], p2[1]]),
                          'u0': corner_angle1,
                          'u1': corner_angle2,
                          'non_manhattan': None})
            if len(walls) >= 2 and walls[-2]['type'] == walls[-1]['type']:
                invalid_manhattan = True
                break            
        else:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            b = p1[1] - m*p1[0]
            non_manhattan = {'m': m, 'b': b}            
            walls.append({'type': 'non_manhattan',
                          'val': 0,
                          'u0': corner_angle1,
                          'u1': corner_angle2,
                          'non_manhattan': non_manhattan})
        if any([idx in curvature_mask for idx in pt_indices]):
            if (p2[0] - p1[0]) == 0:
                non_manhattan = {'m': None, 'b': None}
            else:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                b = p1[1] - m*p1[0]
                non_manhattan = {'m': m, 'b': b}
            walls[-1]['non_manhattan'] = non_manhattan
            
        vi += 1

    # If postprocessing fails, return original polygon
    if invalid_manhattan:
        print('Mostly manhattan postprocessing failed, returning non-manhattan polygon')
        return room_shape
    
    room_shape = mostly_manhattan_shape(walls, [0,0], 0)
    room_shape = np.array(room_shape) + centroid

    return room_shape


def angle_between_vectors(vector1, vector2):
    """Returns the counterclockwise angle between 0 and pi (mirrored about x axis)"""

    unit_v1 = vector1 / np.linalg.norm(vector1)
    unit_v2 = vector2 / np.linalg.norm(vector2)
    rotation_angle = np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1, 1))

    return rotation_angle


def angle_between_vectors2(vector1, vector2):
    """Returns the clockwise angle between 0 and 2pi"""

    init_angle = -np.math.atan2(
        np.linalg.det([vector1, vector2]), np.dot(vector1, vector2)
    )
    rotation_angle = np.mod(
        init_angle + 2 * np.pi, 2 * np.pi
    )

    return rotation_angle


def remove_similar_points(poly, similar_points_threshold=.0075):
    """Remove redundant points which are nearby one one another"""

    poly = poly.tolist()
    vi = 0
    while vi < len(poly):
        pt_indices = [
            (vi - 1) % len(poly),
            (vi) % len(poly)
        ]
        p1, p2 = [poly[idx] for idx in pt_indices]
        pv = np.array(p2) - np.array(p1)

        if np.linalg.norm(pv) < similar_points_threshold:
            new_pt = [np.mean([p1[0], p2[0]]), np.mean([p1[1], p2[1]])]
            skip_pts = pt_indices
            poly = [pt for idx, pt in enumerate(poly) if idx not in skip_pts]
            poly.insert(pt_indices[0], new_pt)
        else:
            vi += 1

    return np.array(poly)


def clean_poly_vertices(poly, colinear_threshold=5, similar_points_threshold=.0075,
                        zig_zag_length_threshold=.1, plot_path=None,
                        remove_zig_zags=True, clean_spikes=False):
    """Merge approximately colinear walls"""

    # Remove points which are nearby one another
    poly = remove_similar_points(poly,
                                 similar_points_threshold=similar_points_threshold)

    poly = poly.tolist()
    changed = True
    while changed:
        changed = False
        vi = 0
        while vi < len(poly):
            pt_indices = [
                (vi - 2) % len(poly),
                (vi - 1) % len(poly),
                (vi) % len(poly),
            ]
            p1, p2, p3  = [poly[idx] for idx in pt_indices]

            pv1 = np.array(p2) - np.array(p1)
            pv2 = np.array(p3) - np.array(p2)

            # Colinear neighbors, delete point
            angle = np.rad2deg(angle_between_vectors(pv1, pv2))
            if angle < colinear_threshold:
                del poly[pt_indices[1]]
                changed = True
                continue
            vi += 1

    # Spikes in the polygon could occur due to noisy segmentation
    # Attempt to clean these
    spike_cleaned = False
    if clean_spikes:
        changed = True
        while changed:
            changed = False
            vi = 0
            while vi < len(poly):
                pt_indices = [
                    (vi - 3) % len(poly),
                    (vi - 2) % len(poly),
                    (vi - 1) % len(poly),
                    (vi) % len(poly),
                    (vi + 1) % len(poly),
                    (vi + 2) % len(poly)
                ]
                p1, p2, p3, p4, p5, p6 = [poly[idx] for idx in pt_indices]

                pv1 = np.array(p2) - np.array(p1)
                pv2 = np.array(p3) - np.array(p2)
                pv3 = np.array(p4) - np.array(p3)
                pv4 = np.array(p5) - np.array(p4)
                pv5 = np.array(p6) - np.array(p5)

                outer_angle1 = np.rad2deg(angle_between_vectors2(pv1, pv2))
                inner_angle = np.rad2deg(angle_between_vectors2(pv2, pv3))
                outer_angle2 = np.rad2deg(angle_between_vectors2(pv3, pv4))
                inner_angle_diff = 180 - inner_angle
                if inner_angle_diff < 5 and inner_angle_diff > 0:
                    del poly[pt_indices[2]]
                    changed = True
                    spike_cleaned = True
                    break

                angle1 = outer_angle1
                angle2 = np.rad2deg(angle_between_vectors2(pv2, pv3))
                angle3 = outer_angle2
                angle4 = np.rad2deg(angle_between_vectors2(pv4, pv5))
                d1 = np.linalg.norm(np.array(p2) - np.array(p3))
                d2 = np.linalg.norm(np.array(p4) - np.array(p5))
                d3 = np.linalg.norm(np.array(p3) - np.array(p4))
                d4 = np.linalg.norm(np.array(p2) - np.array(p5))
                shapely_poly = sg.Polygon([p2, p3, p4, p5])
                poly_area = shapely_poly.area
                dot_prod = (pv2/np.linalg.norm(pv2)).dot(pv4/np.linalg.norm(pv4))

                if angle1 < 180 and angle4 < 180 and \
                   angle2 > 180 and angle3 > 180 and \
                   poly_area < .5 and d1/d3 > 10 and \
                   dot_prod < -.95:

                    if poly_area > .5:
                      print(data_key)
                      print(f'{dot_prod:.3f}, {poly_area:.3f}, {d1/d3:.3f}')

                    skip_indices = pt_indices[2:4]
                    poly = [pt for idx, pt in enumerate(poly) if idx not in skip_indices]
                    changed = True
                    spike_cleaned = True
                    break

                vi += 1

    # If additional nearby points have been created in cleaning spikes, remove them
    poly = remove_similar_points(np.array(poly),
                                 similar_points_threshold=similar_points_threshold)

    return np.array(poly), spike_cleaned


def mask_to_polygon(shape_image, scale, vanish_angle):
    '''
    Run overall postprocessing to convert the segmentation mask into the output
    room shape polygon.
    Args: 
        shape_image: binary image that contains the room shape mask
        scale: image coordinate scale
        vanish_angle:
    Return: 
        Corner coordinates represented with XY
    '''

    # Extract boundary contour(s) from segmentation
    contours, hierarchy = cv2.findContours(shape_image, 3, 2)

    valid_contours = []
    for contour in contours:
        epsilon = 0.005*cv2.arcLength(contour, True)
        if epsilon < 2:
            epsilon = 2
        # print(epsilon)
        contour = contour[:, 0, :]

        # Run contour polygon simplification
        contour = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :]
        valid_contours.append(contour)
    # Retain largest boundary polygon (by area) as input for postprocessing
    valid_contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    merged_room_shape = valid_contours[0]

    # Create vanishing angle rotation matrix and rotate input polygon to
    # align approximately manhattan walls with coordinate axes for mostly
    # manhattan postprocessing
    vanishing_angle_rotation_matrix = np.array(
            [
                [np.cos(vanish_angle), -np.sin(vanish_angle)],
                [np.sin(vanish_angle), np.cos(vanish_angle)],
            ]
        )
    merged_room_shape = merged_room_shape.dot(vanishing_angle_rotation_matrix)

    # Merge approximately co-linear walls prior to postprocessing
    merged_room_shape_clean = clean_poly_vertices(merged_room_shape,
                                                  similar_points_threshold=.1 * scale,
                                                  colinear_threshold=30,
                                                  clean_spikes=True)
    merged_room_shape_clean = merged_room_shape_clean[0]
    plot_path = None

    merged_room_shape_clean = manhattanize_shape(merged_room_shape_clean,length_threshold = 0.01*scale,
                                                 manhattan_threshold=20,
                                                 plot_path=plot_path)

    merged_room_shape_clean = merged_room_shape_clean.dot(vanishing_angle_rotation_matrix.T)

    return merged_room_shape_clean


def rgb_to_gray(image_rgb):
    """Convert a RGB image to a grayscale image.
    Args:
        image_rgb: 3 channel RGB image in the order of [R,G,B]

    Return:
        The corresponding grayscale image.
    """
    if is_grayscale(image_rgb):
        return image_rgb
    else:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def to_float(image_uint8):
    """Convert uint8 image to a floating-point image.
    Args:
        image_uint8: The input uint8 image

    Return:
        Floating-point image.
    """
    if np.issubdtype(image_uint8.dtype, np.floating):
        return image_uint8
    else:
        return image_uint8.astype(np.float32) / 255.0

    
def read_image(file_path, use_float=False):
    image_cv = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if use_float:
        image_cv = to_float(image_cv)

    if len(image_cv.shape) == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    else:
        # Should not happen but just in case.
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)

    return image_cv


class ImageCoordinateAlignment():

    def __init__(self, rotation_matrix=None):
        if rotation_matrix is None:
            self._rotation_matrix = np.array([[1, 0, 0],
                                              [0, 0, -1],
                                              [0, 1, 0]])
        else:
            self._rotation_matrix = rotation_matrix

    def rotate(self, input_array):
        
        return input_array.dot(self._rotation_matrix)
    

def sphere_to_pixel(points_sph, width):
    """Convert spherical coordinates to pixel coordinates inside a 360 pano
    image with a given width.

    Note:
        We assume the width covers the full 360 degrees horizontally, and the
        height is derived as width/2 and covers the full 180 degrees
        vertical, i.e. we support mapping only on full FoV panos.

    Args:
        points_sph: List of points given in spherical coordinates, where
            only the [theta, phi] coordinates are required, thus
            points_sph.shape is (num_points, 2).

        width: The width of the pano image (defines the azimuth scale).

    Return:
        List of points in pano pixel coordinates [x, y], where the spherical
        point [theta=0, phi=0] maps to the image center.
        Shape of the result is (num_points, 2).
    """
    if not isinstance(points_sph, np.ndarray):
        points_sph = np.reshape(points_sph, (1, -1))
        output_shape = (2, )  # type: ignore
    else:
        output_shape = (points_sph.shape[0], 2)  # type: ignore

    num_points = points_sph.shape[0]
    assert(num_points > 0)

    num_coords = points_sph.shape[1]
    assert(num_coords == 2 or num_coords == 3)

    height = width / 2
    assert(width > 1 and height > 1)

    # We only consider the azimuth and elevation angles.
    theta = points_sph[:, 0]


    assert(np.all(np.greater_equal(theta, -math.pi - EPS_RAD)))
    assert(np.all(np.less_equal(theta, math.pi + EPS_RAD)))

    phi = points_sph[:, 1]

    
    assert(np.all(np.greater_equal(phi, -math.pi / 2.0 - EPS_RAD)))
    assert(np.all(np.less_equal(phi, math.pi / 2.0 + EPS_RAD)))

    # Convert the azimuth to x-coordinates in the pano image, where
    # theta = 0 maps to the horizontal center.
    
    x_arr = theta + math.pi  # Map to [0, 2*pi]
    x_arr /= (2.0 * math.pi)  # Map to [0, 1]
    x_arr *= width - 1  # Map to [0, width)
    

    # Convert the elevation to y-coordinates in the pano image, where
    # phi = 0 maps to the vertical center.
    y_arr = phi + math.pi / 2.0  # Map to [0, pi]
    y_arr /= math.pi  # Map to [0, 1]
    y_arr = 1.0 - y_arr  # Flip so that y goes up.
    y_arr *= height - 1  # Map to [0, height)


    return np.column_stack((x_arr, y_arr)).reshape(output_shape)


def cartesian_to_sphere(points_cart,
                        use_unit = False):
    """Convert cartesian to spherical coordinates.

    Args:
        points_cart: List of points given in cartesian coordinates [x, y, z]
            in a row-major order, i.e. first row is [x1, y1, z1], etc.
            points_cars.shape is (num_points, 3)

        use_unit: If set to true, we would assume all points are on the unit
            sphere and we will return only the [theta, phi] part, i.e. the
            shape of the result will be (num_points, 2)

    Return:
        List of points in spherical coordinates: [theta, phi, rho] or
        [theta, phi] (if use_unit is set to true):

        theta is the azimuthal angle in [-pi, pi],
        phi is the elevation angle in [-pi/2, pi/2],
        rho is the radial distance in (0, inf)

        Shape of the result is (num_points, 3) or (num_points, 2) if
        use_unit is set to true
    """
    output_dim = 2 if use_unit else 3
    if not isinstance(points_cart, np.ndarray) or points_cart.ndim == 1:
        points_cart = np.reshape(points_cart, (1, -1))
        output_shape = (output_dim, )
    else:
        output_shape = (points_cart.shape[0], output_dim)  # type: ignore

    num_points = points_cart.shape[0]
    assert (num_points > 0)

    num_coords = points_cart.shape[1]
    assert(num_coords == 3)

    x_arr = points_cart[:, 0]
    y_arr = points_cart[:, 1]
    z_arr = points_cart[:, 2]

    # Azimuth angle is in [-pi, pi]
    theta = np.arctan2(x_arr, -z_arr)

    if use_unit:
        # Note that we can skip the calculation of rho in that case.
        phi = np.arcsin(y_arr)  # Map elevation to [-pi/2, pi/2]
        return np.column_stack((theta, phi)).reshape(output_shape)
    else:
        # Radius can be anything between (0, inf)
        rho = np.sqrt(np.sum(np.square(points_cart), axis=1))
        phi = np.arcsin(y_arr / rho)  # Map elevation to [-pi/2, pi/2]
        return np.column_stack((theta, phi, rho)).reshape(output_shape)

def is_loop_closure_line(pano_width, pt1_cart,
                         pt2_cart):
    """Check if a given line is a "loop closure line", meaning that it's
    rendering on the pano texture would wrap around the left/right border.
    """
    pt1 = np.asarray(pt1_cart).reshape(1, 3)
    pt2 = np.asarray(pt2_cart).reshape(1, 3)

    pt1_pix = sphere_to_pixel(cartesian_to_sphere(pt1), pano_width)
    pt2_pix = sphere_to_pixel(cartesian_to_sphere(pt2), pano_width)

    mid_pt = 0.5 * (pt1 + pt2)
    mid_pt /= np.linalg.norm(mid_pt)

    mid_pt_pix = sphere_to_pixel(cartesian_to_sphere(mid_pt), pano_width)

    dist_total = abs(pt1_pix[0, 0] - pt2_pix[0, 0])
    dist_left = abs(pt1_pix[0, 0] - mid_pt_pix[0, 0])
    dist_right = abs(pt2_pix[0, 0] - mid_pt_pix[0, 0])

    return dist_total > pano_width / 2.0 or \
        dist_left + dist_right > dist_total + 1


def draw_lines(pano_image,
               pt1_list, pt2_list, *,
               color,
               thickness = 1, thresh_deg = 0.5):
    """Draw a spherical line corresponding to the shorter arc, by properly
    handling loop-closure crossing lines.
    """
    thresh_rad = np.deg2rad(thresh_deg)

    width = pano_image.width

    # Draw the images on a copy of the underlying pano image.
    image_lines = pano_image.opencv_image_copy()

    for pt1_cart, pt2_cart in zip(pt1_list, pt2_list):
        pt1 = np.asarray(pt1_cart).reshape(1, 3)
        pt2 = np.asarray(pt2_cart).reshape(1, 3)

        pt1_pix = sphere_to_pixel(cartesian_to_sphere(pt1), width)

        points_stack = [[pt1_pix[0, 0], pt1_pix[0, 1]]]

        lines_stack = [(pt1, pt2)]
        while lines_stack:
            line_curr = lines_stack.pop()

            pt1 = line_curr[0]
            pt2 = line_curr[1]

            dot_curr = np.clip(np.dot(pt1, pt2.T), -1.0, 1.0)
            angle_curr = np.arccos(dot_curr)

            if angle_curr < thresh_rad:
                pt2_pix = sphere_to_pixel(cartesian_to_sphere(pt2), width)
                points_stack.append([pt2_pix[0, 0], pt2_pix[0, 1]])
            else:
                mid_pt = 0.5 * (pt1 + pt2)
                mid_pt /= np.linalg.norm(mid_pt)
                lines_stack.append((mid_pt, pt2))
                lines_stack.append((pt1, mid_pt))

        # In case of a loop closure line, we split it into two poly lines.
        if is_loop_closure_line(width, pt1_cart, pt2_cart):
            idx_cut = -1
            for idx, pt_curr in enumerate(points_stack[:-1]):
                pt_next = points_stack[idx + 1]
                if abs(pt_curr[0] - pt_next[0]) > width / 2:
                    idx_cut = idx

            assert(0 <= idx_cut < len(points_stack))

            points_left = np.int32([points_stack[0:idx_cut + 1]])
            cv2.polylines(image_lines, points_left, False, color, thickness)

            points_right = np.int32([points_stack[idx_cut + 1:-1]])
            cv2.polylines(image_lines, points_right, False, color, thickness)
        else:
            cv2.polylines(image_lines, np.int32([points_stack]),
                          False, color, thickness)

    return PanoImage(image_lines)


def normalize(points_cart):
    """Normalize a set of 3D points/vectors.

    Args:
        points_cart: The list of 3D points/vectors in row-major order.

    Return:
        The normalized vectors (or points lying on the unit sphere).
    """
    num_points = points_cart.shape[0]
    assert (num_points > 0)

    num_coords = points_cart.shape[1]
    assert(num_coords == 3)

    rho = np.sqrt(np.sum(np.square(points_cart), axis=1))
    return points_cart / rho.reshape(num_points, 1)


def draw_room_shape_on_image(floor_coordinates, 
                             ceiling_coordinates,
                             pano_image,
                             color=(0,0,255)):

    # To change coordinate systems for consistency with draw function
    image_coordinate_alignment = ImageCoordinateAlignment()
    for loop_coordinates in [normalize(floor_coordinates), normalize(ceiling_coordinates)]:
        loop_coordinates = image_coordinate_alignment.rotate(loop_coordinates)
        for i in range(loop_coordinates.shape[0]):
            pt1 = loop_coordinates[[i],:].tolist()
            pt2 = loop_coordinates[[(i+1)%floor_coordinates.shape[0]],:].tolist()
            pano_image = draw_lines(pano_image,
                                    pt1,
                                    pt2,
                                    color=color,
                                    thickness=2)

    return pano_image


def floor_map_to_room_shape(room_vertices,
                            ceiling_height,
                            camera_height):
    """Convert room shape polygon to floor and ceiling coordinates"""

    # Extract and format room shape coordinates
    num_vertices = room_vertices.shape[0]
    floor_z = np.repeat([-camera_height], num_vertices).reshape(num_vertices,1)
    ceiling_z = np.repeat([ceiling_height-camera_height], num_vertices).reshape(num_vertices,1)

    # Create floor and ceiling coordinates
    floor_coordinates = np.hstack((room_vertices, floor_z))
    ceiling_coordinates = np.hstack((room_vertices, ceiling_z))


    return floor_coordinates, ceiling_coordinates


class PanoImageException(Exception):
    """Custom exception related to panorama image issues.
    """
    pass


class PanoImage:

    LOG = logging.getLogger(__name__)

    def __init__(self, image_cv):
        """Initialize 360 panorama from a given OpenCV RGB/gray image.

        Args:
           image_cv: uint8 OpenCV image.
        """

        self._image = image_cv

        self._validate_image()

    @classmethod
    def from_file(cls, image_file_path,
                  use_grayscale = False):
        """Initialize 360 panorama from a given RGB image file.

        Args:
            image_file_path: The path to the 360 panorama image.

            use_grayscale: If true we will transform to grayscale
        """
        imread_flags = \
            cv2.IMREAD_GRAYSCALE if use_grayscale else cv2.IMREAD_COLOR

        image_cv = cv2.imread(image_file_path, imread_flags)

        # This can happen because of missing file, improper permissions,
        # unsupported or invalid format, etc.
        if image_cv is None:
            msg = "Can not load image: %s" % image_file_path
            raise PanoImageException(msg)

        # Convert the underlying data structure from uint8 BGR to uint8 RGB
        # if not using a grayscale image.
        if use_grayscale is False:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        return cls(image_cv)

    @classmethod
    def from_width(cls, width: int, *,
                   use_grayscale = False) :
        """Initialize 360 panorama with a given width, the height will be
        calculated as half of the width so that we have a full 360x180 pano.

        Args:
            width: The desired panorama width (in pixels).

            channels: Number of channels, e.g. 1 for grayscale, 3 for RGB

        Return:
            A pano image with the desired dimensions where all pixels
            will be set to 0.
        """
        assert(isinstance(width, int))
        assert(width % 2 == 0)

        height = width // 2  # Integer division.
        if use_grayscale:
            image_cv = np.zeros((height, width), dtype=np.uint8)
        else:
            image_cv = np.zeros((height, width, 3), dtype=np.uint8)

        return cls(image_cv)

    @classmethod
    def from_image(cls, image, working_width, *,
                   use_grayscale = False):
        """Initialize 360 panorama from an existing one with a desired
        working width, i.e. resized.
        """
        assert working_width > 0

        working_height = int(working_width / 2.0)

        image_cv = cv2.resize(image.opencv_image,
                              (working_width, working_height))
        if use_grayscale and not image.is_grayscale:
            image_cv = rgb_to_gray(image_cv)

        return cls(image_cv)

    def _validate_image(self):
        """Verify whether the underlying image represents a valid 360 panorama.
        This method will throw an instance of PanoImageException is this is
        not the case.
        """
        if self._image is None:
            raise PanoImageException("Empty image")

        if not np.issubdtype(self._image.dtype, np.uint8):
            msg = "Expecting uint8 image: %s" % self._image.dtype
            raise PanoImageException(msg)

        # We can use the shape property to check for grayscale images:
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html # noqa
        if not (self.is_grayscale or self.is_rgb):
            msg = "Only grayscale or RGB panos are supported at this moment"
            raise PanoImageException(msg)

        if not self._has_valid_fov:
            msg = "Invalid pano dimensions: %d-by-%d" % \
                  (self.width, self.height)
            raise PanoImageException(msg)

    @property
    def _has_valid_fov(self) -> bool:
        """Return true if the pano image dimensions could represent a valid
        full or limited vertical FoV pano.
        """
        return self.width >= 2 * self.height

    @property
    def width(self) -> int:
        """Return the width of the pano."""
        return self._image.shape[1]  # Get the columns

    @property
    def height(self) -> int:
        """Return the height of the pano."""
        return self._image.shape[0]  # Get the rows

    @property
    def num_channels(self) -> int:
        """Return the number of channels"""
        if len(self._image.shape) < 3:
            return 1
        else:
            return self._image.shape[2]

    @property
    def is_grayscale(self) -> bool:
        """Returns true if the underlying image is grayscale"""
        return self.num_channels == 1

    @property
    def is_rgb(self) -> bool:
        """Returns true if the underlying image is RGB"""
        return self.num_channels == 3

    @property
    def fov(self) -> float:
        """Return the vertical field of view of the pano (in radians)

        Note: Since we support limited FoV panos, e.g. coming from a phone
        capture, this value can be less than Pi, i.e <= 180 degrees.
        """
        # Assuming width always covers 360 (2*Pi), we compute the vertical FoV:
        fov = (self.height * 2 * math.pi) / self.width
        return fov

    @property
    def has_full_fov(self) -> bool:
        """Return true if the pano image covers full vertival FoV (2:1 ratio)
        and false otherwise (for limited FoV panos).
        """
        return self.width == 2 * self.height

    @property
    def opencv_image(self):
        """Return a shallow copy of the underlying OpenCV image data.

        Note: this is a mutable object and should be treated with care!
        """
        return self._image

    def opencv_image_copy(self, *,
                          use_float= False,
                          use_gray= False):

        image_res = self._image.copy()  # This is a deep copy version.

        if use_gray and (not self.is_grayscale):
            image_res = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)

        if use_float:
            image_res = to_float(image_res)

        return image_res

    def resize(self, *, scale: float = 0, working_width: int = 0):

        assert scale > 0 or working_width > 0
        assert not(scale > 0 and working_width > 0)

        if working_width is None and scale is not None:
            # Rond-up the resized width to the closest even number.
            working_width = int(math.ceil(self.width * scale / 2.) * 2)

        height_new = int(working_width / 2.0)

        self._image = cv2.resize(self._image, (working_width, height_new))

    def clear(self):
        """Set all panorama pixels to 0, i.e. generates a black
        """
        self._image[:] = (0, 0, 0)
    def write_to_file(self, file_path: str) -> None:
        # Convert the underlying image from rgb floating point to bgr uint8
        image_bgr_cv = cv2.cvtColor(self.opencv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr_cv)

