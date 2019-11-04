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

import torch.utils.data as data
import pickle
from spconv.utils import VoxelGeneratorV2

class Car3d(data.Dataset):


  def __init__(self, opt, split):
    super(Car3d, self).__init__()
    root_path = opt.root_path
    
    if split =='train':
      info_path = root_path + 'infos_train.pkl'
    elif split == 'val':
      info_path = root_path + 'infos_val.pkl'
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    
    self.infos = data

    self.voxel_generator = VoxelGeneratorV2(
        voxel_size=list(opt.voxel_size),
        point_cloud_range=list(opt.point_cloud_range),
        max_num_points=opt.max_number_of_points_per_voxel,
        max_voxels=20000)
    self.opt = opt
    self.split = split

    print('Loaded {} {} samples'.format(split, len(self.infos)))

  def __len__(self):
    return len(self.infos)

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    pass

  def run_eval(self, results, save_dir):
    pass
