import os
import sys
base_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "sp2"))

import argparse
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import torch
import torch.utils.data as data
from Model.projection import cp2
from Model.e2p import E2P
import pickle as pk
import cv2
import json
from torch.utils.data import DataLoader
import shapely.geometry as sg
import config as cf
from Model.model import PSMNet
from draw_layers import draw_output
import copy
import math
from src.loftr.loftr import LoFTR, lower_config
from src.config.default import get_cfg_defaults
from src.utils.metric import _compute_metrics
from yacs.config import CfgNode as CN
import timeit
from postproc import clean_poly_vertices, mask_to_polygon, PanoImage, read_image, draw_room_shape_on_image, floor_map_to_room_shape


IMAGE_WIDTH = 1024
COARSE_POSE_RATIO = 0.7
COARSE_POSE_NOISE_RATIO = 0.1
# The model was trained with Zillow-internal version of ZInD with different 
# coordinate scaling. Used to be consistent with externally released ZInD.
COORDINATE_SCALE = 0.4042260417272217 
MEDIAN_CAMERA_HEIGHT = 1.45


def predict(config, model, primary_pano_path, secondary_pano_path, json_path, output_instance_vis_path):

    # Load pano images
    pano1 = np.array(Image.open(primary_pano_path).resize((IMAGE_WIDTH,
                                                           IMAGE_WIDTH//2)), np.float32)[..., :3] / 255.
    pano2 = np.array(Image.open(secondary_pano_path).resize((IMAGE_WIDTH,
                                                             IMAGE_WIDTH//2)), np.float32)[..., :3] / 255.

    # Convert pano arrays to tensors
    pano1_tensor = torch.FloatTensor(np.rollaxis(pano1, 2))
    pano1_tensor = pano1_tensor.unsqueeze(0)
    pano2_tensor = torch.FloatTensor(np.rollaxis(pano2, 2))
    pano2_tensor = pano2_tensor.unsqueeze(0)

    # Set model to eval mode
    model.eval()

    # Read input gt data 
    with open(json_path, 'rb') as fh:
        data_dict = json.load(fh)

    # Extract vanishing angle for postprocessing, GT R/t for simulating noisy pose
    vanishing_angle = data_dict["vanishing_angle_pano1"]
    rotation_angle = data_dict['pano2_rotation_relative_to_pano1'] * np.pi/180
    rotation = rotation_angle
    translation = data_dict['pano2_coordinates_relative_to_pano1']

    # Simulate noisy input pose
    coarse_pose_ratio_upper_bound = -config["translation_noise"] * COORDINATE_SCALE / MEDIAN_CAMERA_HEIGHT 
    coarse_pose_ratio_lower_bound = config["translation_noise"] * COORDINATE_SCALE / MEDIAN_CAMERA_HEIGHT
    coarse_pose_data_aug = np.random.uniform(coarse_pose_ratio_lower_bound, coarse_pose_ratio_upper_bound)

    delta_x_fm = -translation[0][0] * COORDINATE_SCALE  
    delta_y_fm = translation[0][1] * COORDINATE_SCALE    
    coarse_delta_x_fm = coarse_pose_data_aug + delta_x_fm
    coarse_delta_y_fm = coarse_pose_data_aug + delta_y_fm

    coarse_pose_ratio_upper_bound_r = -config["rotation_noise"] * np.pi/180
    coarse_pose_ratio_lower_bound_r = config["rotation_noise"] * np.pi/180

    coarse_pose_data_aug_r = np.random.uniform(coarse_pose_ratio_lower_bound_r, coarse_pose_ratio_upper_bound_r)
    coarse_rotation = coarse_pose_data_aug_r + rotation

    # Project pano image textures perspective views
    e2p_m = cp2(
        cf.pano_size, cf.fp_size, cf.fp_fov)
    e2p = E2P(
        cf.pano_size, cf.fp_size, cf.fp_fov
    )
    [ceiling1, floor1] = e2p(pano1_tensor)
    [ceiling2, floor2] = e2p_m(pano2_tensor,
                                delta_x_fm=coarse_delta_x_fm,
                                delta_y_fm=coarse_delta_y_fm,
                                rotation=coarse_rotation) # this should be radian)
    floor1 = torch.FloatTensor(floor1)
    floor2 = torch.FloatTensor(floor2)

    # Format inference input data
    data = {
    "image0": floor1,
    "image1": floor2,
    "coarse_rotation": coarse_rotation,
    "coarse_delta_x_fm": coarse_delta_x_fm,
    "coarse_delta_y_fm": coarse_delta_y_fm,
    "pano2": pano2_tensor,
    "pano1": pano1_tensor
    }

    # Run model inference
    pano1_tensor = pano1_tensor.unsqueeze(0)
    pano2_tensor = pano2_tensor.unsqueeze(0)
    batch = [pano1_tensor, pano2_tensor]
    out = model(batch[:2], data)

    # Convert outputs to segmentation map
    fpmap = torch.sigmoid(out[0])   # Final joint segmentation top-down view
    fcmap1 = torch.sigmoid(out[1])
    fcmap2 = torch.sigmoid(out[2])

    # Pose refinement
    prediction = data['prediction']
    coarse_translation = np.array([[coarse_delta_x_fm, coarse_delta_y_fm]])

    # Format pose prediction, combine input noisy pose with predicted refinement
    pred_delta_x = (data['coarse_delta_x_fm'] + prediction[:, 0]).detach().numpy()
    pred_delta_y = (data['coarse_delta_y_fm'] + prediction[:, 1]).detach().numpy()
    pred_rotation = (data['coarse_rotation'] + prediction[:, 2]).detach().numpy()[0]
    pred_rotation_matrix = np.array(
        [
            [np.cos(pred_rotation), -np.sin(pred_rotation)],
            [np.sin(pred_rotation), np.cos(pred_rotation)],
        ]
    )
    pred_translation = np.array([[pred_delta_x[0], pred_delta_y[0]]])

    # Visualize model segmentation outputs
    draw_output(fcmap1[:, 0:1], 0, os.path.join(output_instance_vis_path, 'pred_segmentation_equi1.jpeg'))
    draw_output(fcmap2[:, 0:1], 0, os.path.join(output_instance_vis_path, 'pred_segmentation_equi2.jpeg'))
    draw_output(fpmap[:, 0:1], 0, os.path.join(output_instance_vis_path, 'pred_segmentation_persp.jpeg'))

    # Scale coordinates for visualization
    image_size = 512
    ceiling_height = data_dict['ceiling_height'] * COORDINATE_SCALE
    ceiling_height_2 = data_dict['ceiling_height'] * COORDINATE_SCALE

    camera_height_2 = COORDINATE_SCALE
    fov_scale = np.tan(np.deg2rad(160//2))*COORDINATE_SCALE
    image_coordinates_scale = image_size//2/fov_scale

    # Convert segmentation outputs to final layout polygon for visualization
    layer_pred = (fpmap[:, 0:1].detach().numpy() > 0.8).astype(np.uint8)
    layer_pred_contour = mask_to_polygon(layer_pred[0,0,:], image_coordinates_scale, vanishing_angle)
    
    layer_pred_contour_norm = (layer_pred_contour - image_size//2) / image_coordinates_scale
    layer_pred_contour_norm[:,1] *= -1
    layer_pred_contour_norm_2 = (layer_pred_contour_norm - pred_translation).dot(pred_rotation_matrix.T)

    # Visualize prediction on image and write to file
    pano_image1 = PanoImage.from_image(PanoImage(read_image(primary_pano_path)), working_width=1024)
    pano_image2 = PanoImage.from_image(PanoImage(read_image(secondary_pano_path)), working_width=1024)
    floor_coordinates, ceiling_coordinates = floor_map_to_room_shape(layer_pred_contour_norm,
                                                                     ceiling_height,
                                                                     COORDINATE_SCALE)
    pano_image1 = draw_room_shape_on_image(
        floor_coordinates,
        ceiling_coordinates,
        pano_image1,
        color=(0,0,255)
    )
    
    image_output_path = os.path.join(output_instance_vis_path,
                                    'pano1_pred_shape.jpeg')
    cv2.imwrite(image_output_path,
            pano_image1.opencv_image[:,:,[2,1,0]])
    floor_coordinates, ceiling_coordinates = floor_map_to_room_shape(layer_pred_contour_norm_2,
                                                                        ceiling_height_2,
                                                                        camera_height_2)
    pano_image2 = draw_room_shape_on_image(floor_coordinates,
                                           ceiling_coordinates,
                                           pano_image2,
                                           color=(0,0,255))

    image_output_path = os.path.join(output_instance_vis_path,
                                    'pano2_pred_shape.jpeg')
    cv2.imwrite(image_output_path,
            pano_image2.opencv_image[:,:,[2,1,0]])

    
def demo():

    # Parse input arguments
    parser = argparse.ArgumentParser(description='PSMNet inference scripts')

    parser.add_argument('--ckpt',  default="checkpoints/model.pth",  
                        help='path to the model ckpt file')
    parser.add_argument('--demo_data',  default="assets",  
                        help='path to the test dataset')
    args_input = parser.parse_args()

    # Set seeds for repeatability
    torch.manual_seed(0)
    np.random.seed(0)

    # Read model config
    args = get_cfg_defaults()
    _args = lower_config(args)
    device = 'cpu'
    config_path = os.path.join(base_dir, "yamls/demo.json")
    assert os.path.exists(config_path)
    with open(config_path, 'r') as fh:
        config = json.load(fh)

    # Create model and load pretrained weights
    model = PSMNet(backbone='resnet18', config_layout=config, config_pose=_args['loftr'])
    model_ckpt_path = args_input.ckpt
    model = torch.load(model_ckpt_path)

    # Format input file paths
    dataset_path = args_input.demo_data
    data_file = [fl for fl in os.listdir(dataset_path) if fl.endswith('.json')][0]
    # Data dict contains GT pose and vanishing angle values
    data_dict_path = os.path.join(dataset_path, data_file)

    data_file_name = data_file.split('.json')[0]
    primary_pano_key = '_'.join(data_file_name.split('_')[2:9])
    secondary_pano_key = '_'.join(data_file_name.split('_')[9:])    
    primary_pano_path = os.path.join(dataset_path, f"{primary_pano_key}.jpg")
    secondary_pano_path = os.path.join(dataset_path, f"{secondary_pano_key}.jpg")

    # Directory to write output visualizations
    output_instance_vis_path = os.path.join(base_dir, "demo_outputs")
    os.makedirs(output_instance_vis_path, exist_ok = True)

    # Run model prediction on inputs
    predict(config, model, primary_pano_path, secondary_pano_path, data_dict_path, output_instance_vis_path)

    print('Inference demo successful')
    
    
if __name__ == '__main__':
    demo()
