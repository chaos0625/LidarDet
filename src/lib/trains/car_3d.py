from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import ddd_decode,_nms,_topk
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process,get_alpha
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from models.utils import _gather_feat, _tranpose_and_gather_feat
import utils.simplevis as simplevis
import cv2

class CarLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CarLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = L1Loss()
    self.crit_rot = BinRotLoss()
    self.opt = opt
  
  def forward(self, outputs, batch):
    opt = self.opt

    hm_loss, dep_loss, rot_loss, dim_loss = 0, 0, 0, 0
    wh_loss, off_loss = 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.dim_weight > 0:
        dim_loss += self.crit_reg(output['dim'], batch['reg_mask'],
                                  batch['ind'], batch['dim']) / opt.num_stacks
      if opt.rot_weight > 0:
        rot_loss += self.crit_rot(output['rot'], batch['rot_mask'],
                                  batch['ind'], batch['rotbin'],
                                  batch['rotres']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['rot_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks
    loss = opt.hm_weight * hm_loss  + \
           opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
            + opt.off_weight * off_loss

    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 
                  'dim_loss': dim_loss, 'rot_loss': rot_loss, 
                  }
    if opt.reg_offset:
      #print("reg loss")
      loss_stats["off_loss"] = off_loss
    return loss, loss_stats

class CarTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CarTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'dim_loss', 'rot_loss']
    if opt.reg_offset:
      loss_states +=['off_loss']
    loss = CarLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    
    batch_size, cat, height, width = output["hm"].size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(output["hm"])
    hm = output['hm']
    rot = output['rot']
    dim = output['dim']
    reg = None
    if opt.reg_offset:
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
    #print(dim)

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

    vis_point_range = [-50, -50, -5, 50, 50, 5]

    for i in range(batch_size):
        gt_boxes = detections[i,:]
        gt_boxes = gt_boxes[gt_boxes[:,-2]>0.2]
        gt_boxes = gt_boxes[:,:-2]
        gt_boxes[:,6] = get_alpha(gt_boxes[:,6:14])
        gt_boxes[:,2] = 0
        #gt_boxes[:,6] = -gt_boxes[:,6] - np.pi/2

        #print(gt_boxes)
        im = simplevis.draw_box_in_bev(batch['img'][i].detach().cpu().numpy(), vis_point_range, gt_boxes, [255, 0, 0], 2)
        cv2.imwrite('../results/%06d.jpg'%iter_id,im)

    """
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = np.zeros((512,512,3),dtype =np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())

      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      #debugger.add_img(img, img_id='out_pred')

      if 1:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)
      """
    pass

  def save_result(self, output, batch, results):
    pass