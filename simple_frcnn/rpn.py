import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from anchor_utils import AnchorGenerator, BoxCoder
from loss_utils import reg_loss
from typing import Dict, Tuple, List, Optional


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head
    输入RPN, 输出roi的RPNhead

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.score_layer = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.reg_layer = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        """
        先只考虑一个feature map的情况
        type: [Tensor] -> Tuple(Tensor,Tensor)
        """
        feature = x
        t = F.relu(self.conv(feature))
        pred_roi_locs = self.reg_layer(t)
        pred_obj_scores = self.score_layer(t)
        return pred_obj_scores, pred_roi_locs


def permute_and_flatten(layer: torch.Tensor, N: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def transform_box_prediction(pred_obj_scores: torch.Tensor, pred_roi_locs: torch.Tensor):
    N, AxC, H, W = pred_obj_scores.shape
    Ax4 = pred_roi_locs.shape[1]
    A = Ax4 // 4
    C = AxC // A
    obj_scores = permute_and_flatten(pred_obj_scores, N, C, H, W)  # (2,22500,1)
    roi_locs = permute_and_flatten(pred_roi_locs, N, 4, H, W)  # (2,22500,4)
    #print(f"pred_obj_scores.shape: {obj_scores.shape}, pred_roi_locs.shape: {roi_locs.shape}")
    return obj_scores, roi_locs


class RegionProposalNetwork(nn.Module):
    """
    rpn实现类

    Args:
          anchor_generator(AnchorGenerator):
          head(nn.Module):rpn预测头
          fg_iou_thresh(float):anchor能被视为有物体的和gt box的iou的最小值
          bg_iou_thresh(float):anchor能被视为无物体的和gt box的iou的最大值
          n_loss_sample(int):计算loss时对anchor box的取样数
          positive_fraction(float):对anchor box取样时正(有物体)anchor的比例
          n_pre_nms(Dict[str,int]):nms之前保留的roi个数，{training:..,testing:..}
          n_post_nms(Dict[str,int]):nms之后保留的roi个数，{training:..,testing:..}
          nms_thresh(float):nms用到的iou阈值
          roi_min_size(float):最小的roi长宽大小
          roi_score_thresh(float):过滤roi需要超过的有物体分数阈值
    """

    def __init__(
            self,
            anchor_generator:AnchorGenerator,
            head: nn.Module,
            # for training loss computation
            fg_iou_thresh: float,
            bg_iou_thresh: float,
            n_loss_sample: int,
            positive_fraction: float,
            # for roi filter
            n_pre_nms: Dict[str, int],
            n_post_nms: Dict[str, int],
            nms_thresh: float,
            roi_min_size: float,
            roi_score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder()
        # for loss computation
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.n_loss_sample = n_loss_sample
        self.positive_fraction = positive_fraction
        # for roi filtering
        self.n_pre_nms = n_pre_nms
        self.n_post_nms = n_post_nms
        self.nms_thresh = nms_thresh
        self.score_thresh = roi_score_thresh
        self.roi_min_size = roi_min_size

    def assign_targets_to_anchors(self,
                                  image_size: torch.Tensor,
                                  anchors: torch.Tensor,  #(N,4)
                                  target_bboxes: torch.Tensor, #(N,4)
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = target_bboxes.device
        # 设置 ahchor box对应的gt label和gt loc
        # 暂时去除超过边框的anchor box
        index_inside = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= image_size[0]) &
            (anchors[:, 3] <= image_size[1])
        )[0]  # [1404,1413..]
        valid_anchor_boxes = anchors[index_inside]

        # 设置锚固框有无物体标签
        # faster rcnn anchor box标签为1的条件：满足以下其一
        # 1 anchor box是某个gt_box和所有anchor_box相交的最大iou
        # 2 anchor box和某个gt_box ious超过0.7
        # anchor box标签为0的条件:不满足上述条件且和任意gt_box iou<0.3
        # 其余anchor box标签为-1

        # 计算gt_box和anchor_boxes的pairwise iou
        ious = torchvision.ops.box_iou(valid_anchor_boxes, target_bboxes)

        # 计算gt_box对应的最大iou锚固框index
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, torch.arange(ious.shape[1])]
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]  # [2262，2269...]

        # np.where的多维度使用比较复杂，以下是示例
        # a=np.array([[1,2],[3,4],[5,6],[3,6]])
        # b=np.array([3,4])
        # print(np.where(a==b))

        # 计算anchor_box对应的最大ious
        anchor_argmax_ious = ious.argmax(axis=1)
        anchor_max_ious = ious[torch.arange(ious.shape[0]), anchor_argmax_ious]  # [0.06811669 0.07083762, ..]

        label = torch.empty((ious.shape[0],), dtype=torch.int32, device=device)
        label.fill_(-1)
        pos_iou_threshold = self.fg_iou_thresh
        neg_iou_threshold = self.bg_iou_thresh
        pos_ratio = self.positive_fraction  # 最大positive label比例
        n_sample = self.n_loss_sample  # 最多激活这么多anchor box
        label[anchor_max_ious < neg_iou_threshold] = 0  # 小于neg阈值的anchor
        label[gt_argmax_ious] = 1  # 最大iou anchor
        label[anchor_max_ious >= pos_iou_threshold] = 1  # 大于pos阈值的anchor


        # 调整一下label的正负标签比例
        # Faster_R-CNN paper: Each mini-batch arises from a single image that contains many positive and negative example anchors,
        # but this will bias towards negative samples as they are dominating.
        # Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch,
        # where the sampled positive and negative anchors have a ratio of up to 1:1.
        # If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones..
        n_pos = int(pos_ratio * n_sample)
        pos_index = torch.where(label == 1)[0]
        if pos_index.shape[0] > n_pos:
            n_disable = pos_index.shape[0]-n_pos
            weights = torch.ones(pos_index.shape[0])
            rand_choice = weights.multinomial(n_disable, replacement=False)
            disable_index = pos_index[rand_choice]
            label[disable_index] = -1
        n_neg = n_sample - torch.sum(label == 1)
        neg_index = torch.where(label == 0)[0]
        if neg_index.shape[0] > n_neg:
            n_disable = neg_index.shape[0] - n_neg
            weights = torch.ones(neg_index.shape[0])
            rand_choice = weights.multinomial(n_disable, replacement=False)
            disable_index = neg_index[rand_choice]
            label[disable_index] = -1

        # 设置锚固框对应的bbox位置，格式为(y_center,x_center,log(h),log(w)) !确定一下这个格式
        max_iou_bbox = target_bboxes[anchor_argmax_ious]  # 每个valid anchor对应的bbox

        anchor_locs = self.box_coder.encode_boxes(valid_anchor_boxes, max_iou_bbox)
        # print(anchor_locs.shape)  # (8940, 4)

        # 输出最终包括无效框的anchor box信息
        anchor_labels = torch.empty((anchors.shape[0],), dtype=torch.int32, device=device)
        anchor_labels.fill_(-1)
        anchor_labels[index_inside] = torch.as_tensor(label, dtype=torch.int32, device=device)
        anchor_locations = torch.empty((anchors.shape[0], 4), dtype=anchor_locs.dtype, device=device)
        anchor_locations.fill_(0.0)
        anchor_locations[index_inside, :] = anchor_locs
        #print(f"anchor_locations.shape: {anchor_locations.shape}, anchor_labels.shape: {anchor_labels.shape}")
        # (22500, 4) (22500,)
        return anchor_locations, anchor_labels

    def assign_batched_targets(self,
                               images: torch.Tensor,
                               anchors: torch.Tensor,  #(N,4)
                               targets: List[Dict[str, torch.Tensor]]
                               #gt_bboxes: List[torch.Tensor],
                               ) -> Tuple[torch.Tensor, torch.Tensor]:

        gt_bboxes = [t["boxes"].to(anchors.dtype) for t in targets]
        gt_roi_locs_list = []
        gt_roi_labels_list = []
        n_images = images.shape[0]
        image_size = images.shape[-2:]
        for i in range(n_images):
            target_bboxes = gt_bboxes[i]
            anchor_locations, anchor_labels = self.assign_targets_to_anchors(image_size,
                                                                             anchors,
                                                                             target_bboxes)
            gt_roi_locs_list.append(anchor_locations.unsqueeze(0))
            gt_roi_labels_list.append(anchor_labels.unsqueeze(0))

        gt_roi_locs = torch.cat(gt_roi_locs_list, 0)  # .view(-1, 4)
        gt_roi_labels = torch.cat(gt_roi_labels_list, 0)  # .view(-1)
        #print(f"gt_roi_labels.shape: {gt_roi_labels.shape}, gt_roi_locs.shape: {gt_roi_locs.shape}")
        # (2,22500) (2,22500,4)
        return gt_roi_locs, gt_roi_labels

    def filter_rpn_proposals(self,
                             pred_rois,  #(2,22500)
                             pred_object_scores,  #(2,22500,4)
                             image_shape):
        #, min_roi_size, n_pre_nms, n_post_nms, nms_thresh):
        """
        过滤rpn产生的roi proposal
        1.过滤掉面积太小的
        2.non max suppression
        """
        final_boxes = []
        final_scores = []
        if self.training:
            n_pre_nms = self.n_pre_nms['training']
            n_post_nms = self.n_post_nms['training']
        else:
            n_pre_nms = self.n_pre_nms['testing']
            n_post_nms = self.n_post_nms['testing']
        for rois, scores in zip(pred_rois, pred_object_scores):
            # 暂时所有图片先都视为统一大小, clip roi到图片边缘
            rois[:, 0:4:2] = torch.clamp(rois[:, 0:4:2], 0, image_shape[0])
            rois[:, 1:4:2] = torch.clamp(rois[:, 1:4:2], 0, image_shape[1])
            # 过滤掉太小的roi
            hs = rois[:, 2] - rois[:, 0]
            ws = rois[:, 3] - rois[:, 1]
            keep = torch.where((hs >= self.roi_min_size) & (ws >= self.roi_min_size))[0]  # 还是对tensor用torch.where吧
            rois = rois[keep, :]
            scores = scores[keep]
            #过滤掉score太小的roi
            keep = torch.where(scores >= self.score_thresh)[0]
            rois, scores = rois[keep, :], scores[keep]
            # 只保留n_pre_nms个rois
            order = scores.argsort(descending=True)[:n_pre_nms]
            rois = rois[order, :]
            scores = scores[order]
            # nms, 只保留n_post_nms个rois
            keep = torchvision.ops.nms(rois, scores, self.nms_thresh)
            keep = keep[:n_post_nms]
            final_scores.append(scores[keep])
            final_boxes.append(rois[keep, :])
        return final_boxes, final_scores

    def compute_loss(self,
                     pred_obj_scores:torch.Tensor,
                     gt_roi_labels:torch.Tensor,
                     pred_roi_locs:torch.Tensor,
                     gt_roi_locs:torch.Tensor,
                     ):
        """
        计算rpn loss
        """
        gt_roi_labels = gt_roi_labels.unsqueeze(-1)  # (2,22500,1)
        #此处gt_roi_labels中已经实现了采样，正为1，负为0，其余-1，仅仅对已采样roi计算loss
        mask_label = (gt_roi_labels != -1)
        rpn_cls_loss = F.binary_cross_entropy_with_logits(pred_obj_scores[mask_label],
                                                          gt_roi_labels[mask_label].float())

        # 只对gt确认是正label的bbox计算边框损失
        mask_reg = (gt_roi_labels > 0).expand_as(pred_roi_locs)  # (2,22500,4)
        mask_loc_preds = pred_roi_locs[mask_reg].view(-1, 4)
        mask_loc_targets = gt_roi_locs[mask_reg].view(-1, 4)

        num_loc = (gt_roi_labels > 0).sum()
        rpn_loc_loss = reg_loss(mask_loc_targets, mask_loc_preds, num_loc)

        rpn_lambda = 10.
        sum_rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
        #print(f"sum_rpn_loss: {sum_rpn_loss}")
        return sum_rpn_loss

    def forward(self,
                images: torch.Tensor,
                features: torch.Tensor,
                targets: Optional[List[Dict[str, torch.Tensor]]] = None,
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            images: 待预测的图片，现在暂时统一大小
            features: 待预测图片的feature_map，现在暂时每个图片统一大小的一张
            targets: 需训练提供，图片中的gt_box位置，形如List[Tensor(N,4)]

        Returns:
            boxes(List[Tensor]):预测好的predicted roi box,形如List[Tensor(N,4)]
            losses(Tensor): 训练则返回rpn sum loss,testing则返回tensor([0.])
        """
        image_shape = images.shape[-2:]

        pred_obj_scores, pred_roi_locs = self.head(features)
        anchors = self.anchor_generator(images, features)
        pred_obj_scores, pred_roi_locs = transform_box_prediction(pred_obj_scores, pred_roi_locs)

        pred_roi_locs = pred_roi_locs.detach()  # tensor (2,22500,4)
        pred_object_scores = pred_obj_scores.flatten(-2).detach()  # tensor (2,22500)
        rois = self.box_coder.decode_boxes(anchors, pred_roi_locs)

        boxes, scores = self.filter_rpn_proposals(rois, pred_object_scores, image_shape)

        rpn_loss = torch.zeros([1])
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            gt_roi_locs, gt_roi_lables = self.assign_batched_targets(images, anchors, targets)
            # (2,22500) (2,22500,4)
            rpn_loss = self.compute_loss(pred_obj_scores, gt_roi_lables, pred_roi_locs, gt_roi_locs)

        return boxes, rpn_loss
