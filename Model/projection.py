import os
import sys
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

CAMERA_HEIGHT_FM = 0.4042260417272217

class cp2(nn.Module):
    def __init__(
            self, equ_size, out_dim, fov, radius=128, up_flip=True):
        super(cp2, self).__init__()

        self.equ_h = equ_size[0]
        self.equ_w = equ_size[1]
        self.out_dim = out_dim
        self.fov = fov
        self.radius = radius
        self.up_flip = up_flip

        R_lst = []
        theta_lst = np.array([-90, 0, 90, 180], np.float32) / 180 * np.pi
        phi_lst = np.array([90, -90], np.float32) / 180 * np.pi

        for theta in theta_lst:
            angle_axis = theta * np.array([0, 1, 0], np.float32)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)

        for phi in phi_lst:
            angle_axis = phi * np.array([1, 0, 0], np.float32)
            R = cv2.Rodrigues(angle_axis)[0]
            R_lst.append(R)

        # R_lst = [Variable(torch.FloatTensor(x)) for x in R_lst]
        R_lst = R_lst[4:]
        self.register_buffer('R_lst', torch.FloatTensor(R_lst))

        #equ_cx = (self.equ_w - 1) / 2.0
        #equ_cy = (spelf.equ_h - 1) / 2.0
        self.c_x_orig = (self.out_dim - 1) / 2.0
        self.c_y_orig = (self.out_dim - 1) / 2.0

        self.wangle = (180 - self.fov) / 2.0
        w_len = 2 * self.radius * np.sin(np.radians(self.fov / 2.0)) / np.sin(np.radians(self.wangle))
        self.interval = w_len / (self.out_dim - 1)
        self.ratio_geometry_to_pixel = (self.out_dim - 1) / (2 * self.radius * np.sin(np.radians(self.fov / 2.0)))
        self.register_buffer('z_map', torch.zeros([self.out_dim, self.out_dim]) + self.radius)


    def forward(self, batch, delta_x_fm=0, delta_y_fm=0, rotation=0):

        input_dtype = batch.dtype
        input_device = batch.device

        #with torch.no_grad():
        delta_x = delta_x_fm / CAMERA_HEIGHT_FM * self.radius * np.sin(np.radians(self.wangle)) * self.ratio_geometry_to_pixel;
        delta_y = delta_y_fm / CAMERA_HEIGHT_FM * self.radius * np.sin(np.radians(self.wangle)) * self.ratio_geometry_to_pixel;
        c_x = delta_x + self.c_x_orig
        c_y = delta_y + self.c_y_orig



        x_map_orig = self.interval*(torch.arange(self.out_dim, dtype=input_dtype, device=input_device) - c_x).repeat(self.out_dim, 1)
        y_map_orig = self.interval*(torch.arange(self.out_dim, dtype=input_dtype, device=input_device) - c_y).repeat(self.out_dim, 1).T

        x_map = x_map_orig * np.cos(rotation) - y_map_orig * np.sin(rotation)
        y_map = x_map_orig * np.sin(rotation) + y_map_orig * np.cos(rotation)

        D = torch.sqrt(x_map**2 + y_map**2 + self.z_map**2)

        xyz = torch.stack([x_map, y_map, self.z_map], dim=2)*(self.radius / D.unsqueeze(2).expand(-1, -1, 3))

        reshape_xyz = xyz.view(self.out_dim * self.out_dim, 3).transpose(0, 1)

        loc = []

        R = self.R_lst[0]
        result = torch.matmul(R, reshape_xyz).transpose(0, 1)

        lon = torch.atan2(result[:, 0] , result[:, 2]).view(1, self.out_dim, self.out_dim, 1) / np.pi
        lat = torch.asin(result[:, 1] / self.radius).view(1, self.out_dim, self.out_dim, 1) / (np.pi / 2)

        loc.append(torch.cat([lon, lat], dim=3))

        c_x = delta_x + self.c_x_orig
        c_y = -delta_y + self.c_y_orig

        x_map_orig = self.interval*(torch.arange(self.out_dim, dtype=input_dtype, device=input_device) - c_x).repeat(self.out_dim, 1)
        y_map_orig = self.interval*(torch.arange(self.out_dim, dtype=input_dtype, device=input_device) - c_y).repeat(self.out_dim, 1).T

        x_map = x_map_orig * np.cos(-rotation) - y_map_orig * np.sin(-rotation)
        y_map = x_map_orig * np.sin(-rotation) + y_map_orig * np.cos(-rotation)

        D = torch.sqrt(x_map**2 + y_map**2 + self.z_map**2)

        xyz = torch.stack([x_map, y_map, self.z_map], dim=2)*(self.radius / D.unsqueeze(2).expand(-1, -1, 3))



        reshape_xyz = xyz.view(self.out_dim * self.out_dim, 3).transpose(0, 1)
        R = self.R_lst[1]
        result = torch.matmul(R, reshape_xyz).transpose(0, 1)

        lon = torch.atan2(result[:, 0] , result[:, 2]).view(1, self.out_dim, self.out_dim, 1) / np.pi
        lat = torch.asin(result[:, 1] / self.radius).view(1, self.out_dim, self.out_dim, 1) / (np.pi / 2)

        loc.append(torch.cat([lon, lat], dim=3))

        up_coor, down_coor = loc

        batch = batch.float()
        up_view = F.grid_sample(batch[0:1], up_coor, align_corners=False)
        down_view = F.grid_sample(batch[0:1], down_coor, align_corners=False)
        if self.up_flip:
            up_view = torch.flip(up_view, dims=[2])

        # return up_views, down_views
        return up_view, down_view

    def GetGrid(self):
        return self.xyz

