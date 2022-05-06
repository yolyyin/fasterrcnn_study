from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np


class AnchorGenerator(nn.Module):
    """
    为某大小的feature map生成anchor boxes,
    暂时只考虑统一大小的图片缩放到统一大小的单张feature maps
    需要输入一组anchor_scales和ratios的组合，
    以feature map每一个像素点为网格中心生成len(anchor_scales)*len(ratios)个anchor boxes
    输出anchor boxes矩阵，形状(N, 4)

    Args:
        anchor_scales (Tuple[float]):
        ratios (Tuple[float]):
        sub_sample (int):
    """

    def __init__(self,
                 anchor_scales=(8., 16., 32.),
                 ratios=(0.5, 1., 2.),
                 ):
        super(AnchorGenerator, self).__init__()

        self.feature_scale_down = None
        self.anchor_scales = anchor_scales
        self.ratios = ratios
        self.anchor_base = None

    def generate_anchors(self,
                         scales: Tuple[float,...],
                         ratios: Tuple[float,...],
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device('cpu'),
                         ):
        # generate anchor box
        hw_pairs = [(self.feature_scale_down[0] * a_s * np.sqrt(r), self.feature_scale_down[1] * a_s * np.sqrt(1. / r))
                    for r in ratios for a_s in scales]
        hw_pairs = torch.as_tensor(hw_pairs, dtype=dtype, device=device)
        anchor_base = torch.hstack((hw_pairs * (-1), hw_pairs)) / 2
        return anchor_base.round()

    # 为了保证dtype和device冗余地把self.anchor_base再设置一下
    def set_anchor_base(self, dtype: torch.dtype, device: torch.device):
        self.anchor_base = self.anchor_base.to(dtype=dtype, device=device)

    def num_anchors_per_location(self):
        return len(self.anchor_scales)*len(self.ratios)

    def grid_anchors(self, grid_size: List[int], strides: List[int], device) -> torch.Tensor:
        # 计算原图所有anchor box的四角坐标
        grid_width, grid_height = grid_size[0], grid_size[1]
        # 计算原图网格中心点
        shifts_x = torch.arange(0.5, grid_width + 0.5, dtype=torch.float32, device=device) * strides[0]
        shifts_y = torch.arange(0.5, grid_height + 0.5, dtype=torch.float32, device=device) * strides[1]
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        # flatten
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack((shifts_y, shifts_x, shifts_y, shifts_x), dim=1)
        # 排列组合broadcasting
        anchors = (shifts.view(-1, 1, 4) + torch.as_tensor(self.anchor_base, device=device).view(1, -1, 4)).reshape(-1, 4)
        # print(anchors.shape)  # torch (22500,4)
        return anchors

    def forward(self, images: torch.Tensor, features: torch.Tensor):
        grid_size = features.shape[-2:]
        image_size = images.shape[-2:]
        dtype, device = features.dtype, features.device
        strides = [
            image_size[0] // grid_size[0],
            image_size[1] // grid_size[1],
        ]
        self.feature_scale_down = strides
        self.anchor_base = self.generate_anchors(self.anchor_scales, self.ratios, dtype, device)
        self.set_anchor_base(dtype, device)
        anchors = self.grid_anchors(grid_size, strides, device)
        return anchors  # (22500,4)


class BoxCoder:
    '''
    此类参考参考box为bbox编解码
    '''

    def encode_boxes(self, anchors, bboxes):  # input (N,4) (M,4)
        """
        输入anchors列表和bboxes列表，输出bboxes相对anchors编码后的值
        :param anchors:
        :param bboxes:
        :return: 编码后的bboxes值列表
        """
        a_height = anchors[:, 2] - anchors[:, 0]
        a_width = anchors[:, 3] - anchors[:, 1]
        a_ctr_y = anchors[:, 0] + 0.5 * a_height
        a_ctr_x = anchors[:, 1] + 0.5 * a_width
        base_height = bboxes[:, 2] - bboxes[:, 0]
        base_width = bboxes[:, 3] - bboxes[:, 1]
        base_ctr_y = bboxes[:, 0] + 0.5 * base_height
        base_ctr_x = bboxes[:, 1] + 0.5 * base_width

        dy = (base_ctr_y - a_ctr_y) / a_height
        dx = (base_ctr_x - a_ctr_x) / a_width
        dh = torch.log(base_height / a_height)
        dw = torch.log(base_width / a_width)
        encoded_locs = torch.vstack((dy, dx, dh, dw)).transpose(0, 1)
        return encoded_locs

    def decode_boxes(self, anchors: torch.Tensor, roi_locs: torch.Tensor) -> torch.Tensor:
        '''
        写的很verbose，暂且这样啦
        '''
        # 多张图片维度大于2的情况
        saved_dim = roi_locs.dim()
        if saved_dim > 2:
            a_height = anchors[:, 2] - anchors[:, 0]
            a_width = anchors[:, 3] - anchors[:, 1]
            a_ctr_y = anchors[:, 0] + 0.5 * a_height
            a_ctr_x = anchors[:, 1] + 0.5 * a_width
            dy = roi_locs[:, :, 0::4]  # 这句话还是比较奇怪的，可能考虑bbox不一定是(-1,4)大，也可能(-1,A*4)大
            dx = roi_locs[:, :, 1::4]
            dh = roi_locs[:, :, 2::4]
            dw = roi_locs[:, :, 3::4]
            pred_ctr_x = dx * a_width[None, :, None] + a_ctr_x[None, :, None]
            pred_ctr_y = dy * a_height[None, :, None] + a_ctr_y[None, :, None]
            pred_w = torch.exp(dw) * a_width[None, :, None]
            pred_h = torch.exp(dh) * a_height[None, :, None]
            pred_boxes1 = pred_ctr_y - 0.5 * pred_h
            pred_boxes2 = pred_ctr_x - 0.5 * pred_w
            pred_boxes3 = pred_ctr_y + 0.5 * pred_h
            pred_boxes4 = pred_ctr_x + 0.5 * pred_w
            pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=3).flatten(2)
        else:
            a_height = anchors[:, 2] - anchors[:, 0]
            a_width = anchors[:, 3] - anchors[:, 1]
            a_ctr_y = anchors[:, 0] + 0.5 * a_height
            a_ctr_x = anchors[:, 1] + 0.5 * a_width
            dy = roi_locs[:, 0::4]  # 这句话还是比较奇怪的，可能考虑bbox不一定是(-1,4)大，也可能(-1,A*4)大
            dx = roi_locs[:, 1::4]
            dh = roi_locs[:, 2::4]
            dw = roi_locs[:, 3::4]
            pred_ctr_x = dx * a_width[:, None] + a_ctr_x[:, None]
            pred_ctr_y = dy * a_height[:, None] + a_ctr_y[:, None]
            pred_w = torch.exp(dw) * a_width[:, None]
            pred_h = torch.exp(dh) * a_height[:, None]
            pred_boxes1 = pred_ctr_y - 0.5 * pred_h
            pred_boxes2 = pred_ctr_x - 0.5 * pred_w
            pred_boxes3 = pred_ctr_y + 0.5 * pred_h
            pred_boxes4 = pred_ctr_x + 0.5 * pred_w
            pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

        return pred_boxes  # (128,84)