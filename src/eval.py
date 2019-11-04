import _init_paths

from opts import opts
import utils.simplevis as simplevis
from spconv.utils import VoxelGeneratorV2
from models.model import create_model, load_model, save_model,voxelnet
from models.networks.dlav0 import get_pose_net
from models.utils import _sigmoid
import utils.simplevis as simplevis
import utils.box_np_ops as box_np_ops
from datasets.dataset_factory import get_dataset
from models.decode import ddd_decode,_nms,_topk
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process,get_alpha
from utils.oracle_utils import gen_oracle_map
from models.utils import _gather_feat, _tranpose_and_gather_feat
from trains.base_trainer import example_convert_to_torch
import pypcd

import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import pickle
import cv2
import time
import onnx
import inspect

def bev_box_overlap(boxes, qboxes, criterion=-1, stable=True):
    riou = box_np_ops.riou_cc(boxes, qboxes)
    return riou

def load_pcd(point_cloud_path):
    pc_data = pypcd.point_cloud_from_path(point_cloud_path).pc_data
    print("point cloud loaded")

    pts = np.zeros([pc_data.size,3],dtype=np.float32)
    print(pc_data.size)

    return pts

def eval(opt):
    Dataset = get_dataset('car3d', 'cardet')
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    voxel_generator = VoxelGeneratorV2(
            voxel_size=list(opt.voxel_size),
            point_cloud_range=list(opt.point_cloud_range),
            max_num_points=opt.max_number_of_points_per_voxel,
            max_voxels=20000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = '/home/zhaozhe/packages/CenterNet/exp/cardet/car/model_last.pth'

    grid_size = voxel_generator.grid_size
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range

    net = voxelnet(opt).to(device)
    net = load_model(net, ckpt_path)
    net.eval()

    with open('/data/object3d/infos_val.pkl', 'rb') as f:
        data = pickle.load(f)
    infos = data

    pp_sum = 0
    np_sum = 0
    pp_sum2 = 0
    pn_sum = 0

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    for k in range(len(infos)):

        info = infos[k]
        lidar_path =  str(info['lidar_path'])

        points = np.fromfile(
            lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])[:,:3]

        res = voxel_generator.generate(
            points, 25000)

        voxels = res["voxels"]
        coords = res["coordinates"]
        num_points = res["num_points_per_voxel"]

        #print('anchors',anchors.shape)
        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.float32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.float32, device=device)

        output = net(voxels,coords,num_points)[-1]

        output['hm'] = output['hm'].sigmoid_()
        batch_size, cat, height, width = output["hm"].size()

        heat = _nms(output["hm"])
        hm = output['hm']
        rot = output['rot']
        dim = output['dim']
        reg = output['reg']
        K = opt.K

        scores, inds, clses, ys, xs = _topk(heat, K=opt.K)
        
        rot = _tranpose_and_gather_feat(rot, inds)
        rot = rot.view(batch_size, K, 8)
        dim = _tranpose_and_gather_feat(dim, inds)
        dim = dim.view(batch_size, K, 3)
        clses  = clses.view(batch_size, K, 1).float()
        scores = scores.view(batch_size, K, 1)
        xs = xs.view(batch_size, K, 1)
        ys = ys.view(batch_size, K, 1)

        xs = xs*opt.down_ratio*opt.voxel_size[0] + opt.point_cloud_range[0]
        ys = ys*opt.down_ratio*opt.voxel_size[1] + opt.point_cloud_range[1]

        if reg is not None:
          reg = _tranpose_and_gather_feat(reg, inds)
          reg = reg.view(batch_size, K, 2)
          xs = xs.view(batch_size, K, 1) + reg[:, :, 0:1]
          ys = ys.view(batch_size, K, 1) + reg[:, :, 1:2]
        

        detections = torch.cat(
              [xs, ys, xs, dim,rot,scores,clses], dim=2)
        detections = detections.detach().cpu().numpy()
        vis_voxel_size = [0.1, 0.1, 0.1]
        vis_point_range = [-50, -50, -5, 50, 50, 5]

        gt_boxes = info["gt_boxes"]
        gt_boxes = gt_boxes[(info["gt_names"]=='Car')&(gt_boxes[:,0]>pc_range[0])&(gt_boxes[:,0]<pc_range[3])&(gt_boxes[:,1]>pc_range[1])&(gt_boxes[:,1]<pc_range[4])]
        point_counts = box_np_ops.points_count_rbbox(points, gt_boxes)
        gt_boxes = gt_boxes[point_counts>10]

        bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
        bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, gt_boxes, [0, 255, 0], 2)

        pred_boxes = detections[0,:]
        pred_boxes = pred_boxes[pred_boxes[:,-2]>0.2]
        pred_boxes = pred_boxes[:,:-2]
        pred_boxes[:,6] = get_alpha(pred_boxes[:,6:14])
        pred_boxes[:,2] = 0

        bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, pred_boxes, [255, 0, 0], 2)
        cv2.imwrite('../results/%06d.jpg'%k,bev_map)

        pp_num = 0
        np_num = 0
        overlaps = bev_box_overlap(gt_boxes[:,[0,1,3,4,6]],
                                           pred_boxes[:,[0,1,3,4,6]]).astype(np.float64)

        for i in range(gt_boxes.shape[0]):

            det_idx = -1
            min_overlap = 0.5

            for j in range(pred_boxes.shape[0]):

                overlap = overlaps[i, j]

                if (overlap > min_overlap):
                    det_idx = j

            if (det_idx == -1):
                np_num += 1
            else :
                # only a tp add a threshold.
                pp_num += 1

        pp_sum +=pp_num
        np_sum +=np_num

        pp_num = 0
        pn_num = 0

        for i in range(pred_boxes.shape[0]):

            det_idx = -1
            min_overlap = 0.5

            for j in range(gt_boxes.shape[0]):


                overlap = overlaps[j, i]

                if (overlap > min_overlap):
                    det_idx = j

            if (det_idx == -1):
                pn_num += 1
            else:
                pp_num +=1

        pp_sum2 +=pp_num
        pn_sum +=pn_num


    print("recall ",pp_sum*1.0/(pp_sum+np_sum))
    print("precision ",pp_sum2*1.0/(pp_sum2+pn_sum))

if __name__ == '__main__':
  opt = opts().parse()
  eval(opt)
