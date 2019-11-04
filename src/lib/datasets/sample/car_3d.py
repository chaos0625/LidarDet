from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import pycocotools.coco as coco
from pathlib import Path
import utils.simplevis as simplevis
import utils.box_np_ops as box_np_ops

class CarDataset(data.Dataset):

  def _convert_alpha(self, alpha):
    return alpha

  def __getitem__(self, index):

    info = self.infos[index]

    lidar_path = Path(info['lidar_path'])
    points = np.fromfile(
        str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])[:,:3]


    inputs = self.voxel_generator.generate(
        points, 25000)

    grid_size = self.voxel_generator.grid_size
    voxel_size = self.voxel_generator.voxel_size
    pc_range = self.voxel_generator.point_cloud_range

    height, width = grid_size[0], grid_size[1]

    self.opt.output_h = height// self.opt.down_ratio
    self.opt.output_w = width // self.opt.down_ratio
    self.max_objs = 200
    

    hm = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    rotbin = np.zeros((self.max_objs, 2), dtype=np.int64)
    rotres = np.zeros((self.max_objs, 2), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

    gt_boxes = info["gt_boxes"]
    gt_boxes = gt_boxes[(info['gt_names']=='Car')&(gt_boxes[:,0]>pc_range[0])&(gt_boxes[:,0]<pc_range[3])&(gt_boxes[:,1]>pc_range[1])&(gt_boxes[:,1]<pc_range[4])]
    point_counts = box_np_ops.points_count_rbbox(points, gt_boxes)
    gt_boxes = gt_boxes[point_counts>10]


    num_objs = min(len(gt_boxes), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -50, -5, 50, 50, 5]

    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, gt_boxes, [0, 255, 0], 2)
    #cv2.imshow("image", bev_map)
    #cv2.waitKey(0)
    #print(gt_boxes[:,6].min(),gt_boxes[:,6].max())

    for k in range(len(gt_boxes)):

      cls_id = 0

      gt_box = gt_boxes[k]

      ct_x = int((gt_box[0] - pc_range[0])/voxel_size[0])/self.opt.down_ratio
      ct_y = int((gt_box[1] - pc_range[1])/voxel_size[1])/self.opt.down_ratio

      w = gt_box[3]/voxel_size[0]/self.opt.down_ratio
      h = gt_box[4]/voxel_size[1]/self.opt.down_ratio

      ra = max(w,h)
      radius = gaussian_radius((ra, ra))
      radius = max(0, int(radius))

      ct = np.array([ct_x,ct_y],dtype = np.float32)
      ct_int = ct.astype(np.int32)
      draw_gaussian(hm[cls_id], ct, radius)

      if 1:
        alpha = self._convert_alpha(gt_box[6])
        # print('img_id cls_id alpha rot_y', img_path, cls_id, alpha, ann['rotation_y'])
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
          rotbin[k, 0] = 1
          rotres[k, 0] = alpha - (-0.5 * np.pi)    
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
          rotbin[k, 1] = 1
          rotres[k, 1] = alpha - (0.5 * np.pi)
        dim[k] = gt_box[3:6]
        # print('        cat dim', cls_id, dim[k])
        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg[k] = gt_box[:2] - ct_int*self.opt.down_ratio*voxel_size[:2] - pc_range[:2]
        reg_mask[k] = 1 
        rot_mask[k] = 1

    voxels = inputs["voxels"]
    coordinates = inputs["coordinates"]
    num_points = inputs["num_points_per_voxel"]


    ret = {'img':bev_map,'voxels': voxels,"coors":coordinates,"num_points":num_points, 
           'hm': hm, 'dim': dim, 'ind': ind, 
           'rotbin': rotbin, 'rotres': rotres, 'reg_mask': reg_mask,
           'rot_mask': rot_mask}
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret
