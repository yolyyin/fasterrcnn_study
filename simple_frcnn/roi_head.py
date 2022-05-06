import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.ops as ops
from typing import List,Dict,Tuple

from anchor_utils import BoxCoder
from loss_utils import reg_loss


class ROI_pool(nn.Module):
    def __init__(self, output_size):
        super(ROI_pool, self).__init__()
        self.out_size = output_size
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self,
                features,  # type tensor  (num_image,C,H,W)
                cat_rois,  # type tensor  (n_rois,5)
                ):
        output = []
        cat_rois[:, 1:].mul_(1 / 16)
        cat_rois = torch.round(cat_rois).long()
        for i, roi in enumerate(cat_rois):
            im_idx = int(roi[0].item())
            patch = features.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
            output.append(self.adaptive_max_pool(patch))
        output = torch.cat(output, dim=0)
        return output


# frcnn fc classifier
class FRCNN_classifier(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(FRCNN_classifier, self).__init__()
        self.in_channel = input_channel
        self.n_classes = num_classes
        self.roi_head_classifier = nn.Sequential(nn.Linear(self.in_channel, 4096),
                                                 nn.ReLU(),
                                                 nn.Linear(4096, 4096),
                                                 nn.ReLU())
        self.loc_classifier = nn.Linear(4096, self.n_classes*4)
        self.score_classifier = nn.Linear(4096, self.n_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.roi_head_classifier(x)
        cls_loc = self.loc_classifier(x)
        cls_score = self.score_classifier(x)
        return cls_loc, cls_score


class RoIHeads(nn.Module):

    def __init__(self,
                 box_roi_pool,
                 box_predictor,
                 # Faster rcnn traing
                 fg_iou_thresh,
                 bg_iou_thresh_hi,
                 bg_iou_thresh_lo,
                 n_loss_sample,
                 positive_fraction,
                 # Faster rcnn inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 ):
        super().__init__()
        self.box_coder = BoxCoder()

        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh_hi = bg_iou_thresh_hi
        self.bg_iou_thresh_lo = bg_iou_thresh_lo
        self.n_loss_sample = n_loss_sample
        self.positive_fraction = positive_fraction

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def select_training_samples(self,
                                rois,
                                targets,  # List[Dict[str,Tensor]]
                                ):
        """
        ！！！整个写的verbose，继续修改
        改变target形式到roi，并随机选出一些roi pos(有物体框）,neg（背景） sample，其余的roi全部擦去
        输出： list(tensor), list(tensor), list(tensor)
        """
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = rois[0].dtype
        device = rois[0].device

        gt_bboxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        target_regressions = []
        target_labels = []
        roi_samples = []
        for rois_per_image, bboxes_per_image, labels_per_image in zip(rois, gt_bboxes, gt_labels):
            if bboxes_per_image.numel() == 0:  # 防止图片中没有物体框的情况
                # !!!感觉好像可以这样跳过，不计入target和roi_samples，存疑
                continue

            ious = torchvision.ops.box_iou(rois_per_image, bboxes_per_image)

            gt_assignment = ious.argmax(axis=1)  # 每个roi对应的bbox序号
            max_iou, _ = ious.max(axis=1)
            gt_roi_label = labels_per_image[gt_assignment]  # 每个roi对应的label

            num_pos = int(self.n_loss_sample * self.positive_fraction)
            pos_index = torch.where(max_iou >= self.fg_iou_thresh)[0]
            pos_this_image = int(min(num_pos, pos_index.shape[0]))
            if pos_this_image > 0:
                weights = torch.ones(pos_index.shape[0])
                rand_choice = weights.multinomial(pos_this_image, replacement=False)
                pos_index = pos_index[rand_choice]

            neg_index = torch.where((max_iou < self.bg_iou_thresh_hi) & (max_iou >= self.bg_iou_thresh_lo))[0]
            neg_this_image = int(min(self.n_loss_sample - pos_this_image, neg_index.shape[0]))
            if neg_this_image > 0:
                weights = torch.ones(neg_index.shape[0])
                rand_choice = weights.multinomial(neg_this_image, replacement=False)
                neg_index = neg_index[rand_choice]
            keep_index = torch.cat((pos_index, neg_index))
            gt_roi_label = gt_roi_label[keep_index]
            gt_roi_label[pos_this_image:] = 0
            sample_rois = rois_per_image[keep_index, :]

            bboxes_for_sampled_roi = bboxes_per_image[gt_assignment[keep_index]]
            bboxes_attach_roi = self.box_coder.encode_boxes(sample_rois, bboxes_for_sampled_roi)

            target_regressions.append(bboxes_attach_roi)
            target_labels.append(gt_roi_label)
            roi_samples.append(sample_rois)
        return roi_samples, target_labels, target_regressions

    def convert_to_roi_format(self, boxes: List[torch.Tensor]) -> torch.Tensor:
        """
        把tensor形式的roi转化成符合roi_pool输入要求的带有图片ids的（N，5）形状
        这个地方传入的roi是List[Tensor]形式，后面postprocess_detection有用
        """
        concat_rois = torch.cat(boxes, dim=0)
        device, dtype = concat_rois.device, concat_rois.dtype
        ids = torch.cat([
            torch.full_like(b[:, :1], i, dtype=dtype, device=device)
            for i, b in enumerate(boxes)
        ], dim=0)
        concat_rois = torch.cat([ids, concat_rois], dim=1)
        return concat_rois  #(256，5)



    def postprocess_detections(self,
                               class_logits,  # type Tensor (256,21)
                               box_regression,  # type Tensor (256,84)
                               roi_proposals,  # type list[Tensor]
                               image_shape,  # type Tuple[int,int]
                               ): #() -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        n_classes = class_logits.shape[-1]

        # 改变一下regression和logits的数据形式
        pred_boxes_encoded = box_regression  # (256,84)
        pred_scores = F.softmax(class_logits, -1)

        # 分开之前concat的boxes到各个图片
        boxes_per_image = [roi_in_image.shape[0] for roi_in_image in roi_proposals]
        boxes_enconded_list = pred_boxes_encoded.split(boxes_per_image, 0)
        scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        # decode_boxes(anchors: torch.Tensor, roi_locs: torch.Tensor) 单张图片上
        for boxes_encoded, scores, roi in zip(boxes_enconded_list, scores_list, roi_proposals):
            boxes = self.box_coder.decode_boxes(roi, boxes_encoded)  # (128, 84)
            boxes = boxes.view(-1, n_classes, 4)  # (128,21,4)
            boxes = ops.clip_boxes_to_image(boxes.view(-1, 4), image_shape).view(-1, n_classes, 4)  # （128,21,4）

            # scores (128,21)
            # 创建label Tensor
            labels = torch.arange(n_classes, device=device)  # (1,21) 0到20
            labels = labels.view(1, -1).expand_as(scores)  # (128,21)

            # 去除背景预测
            boxes = boxes[:, 1:, :]  # (128,20,4)
            scores = scores[:, 1:]  # (128,20)
            labels = labels[:, 1:]  # (128,20)

            # 重新concat batch一下，因为有label在所以可以分辨哪个box是哪个物体类
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 去除低分框
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 去除空框
            keep = ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # nms suppression, 每个类独立地做，由batched_nms实现
            keep = ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # 最多保留detection_per_image个
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def compute_loss(self,
                     frcnn_cls_score: torch.Tensor,
                     target_rcnn_labels: torch.Tensor,
                     frcnn_cls_loc: torch.Tensor,
                     target_rcnn_regs: torch.Tensor,
                     ) -> torch.Tensor:
        frcnn_cls_loss = F.cross_entropy(frcnn_cls_score, target_rcnn_labels, ignore_index=-1)

        # 找来对的类对应的边框
        n_sample = frcnn_cls_loc.shape[0]
        frcnn_cls_loc = frcnn_cls_loc.view(n_sample, -1, 4)
        # print(frcnn_cls_loss)
        valid_loc = frcnn_cls_loc[torch.arange(0, n_sample), target_rcnn_labels]
        # 只对gt确认是正label的bbox计算边框损失
        mask = (target_rcnn_labels > 0).unsqueeze(-1).expand_as(valid_loc)
        valid_loc = valid_loc[mask].view(-1, 4)
        target_loc = target_rcnn_regs[mask].view(-1, 4)

        n_box = valid_loc.shape[0]
        frcnn_loc_loss = reg_loss(target_loc, valid_loc, n_box)
        # print(frcnn_loc_loss)
        rcnn_lambda = 10.
        sum_frcnn_loss = frcnn_cls_loss + (rcnn_lambda * frcnn_loc_loss)
        #print(f"sum_frcnn_loss:{sum_frcnn_loss}")
        return sum_frcnn_loss


    def forward(self,
                features,  # type torch.Tensor,
                roi_proposals,  # type List[torch.Tensor]
                image_shape,  # type Tuple[int,int]
                targets = None,  # type Optional[List[Dict[str, Tensor]]]
                ):  # -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """

        Args:
            features (Tensor)
            roi_proposals (List[Tensor[N, 4]])
            image_shapes (Tuple[H, W])
            targets (List[Dict])
        """

        ##if targets is not None:
            ##此处应该有一段检查targets格式的代码

        if self.training:
            roi_proposals, target_labels, target_regressions = self.select_training_samples(roi_proposals, targets)
            target_labels = torch.cat(target_labels, dim=0)
            target_regressions = torch.cat(target_regressions, dim=0)
            #print(target_labels.shape, target_regressions.shape)  # （256） （256，4）
        else:
            target_labels = None
            target_regressions = None

        concat_rois = self.convert_to_roi_format(roi_proposals)
        #print(concat_rois.shape)  # (256，5)
        box_features = self.box_roi_pool(features, concat_rois)
        frcnn_cls_loc, frcnn_cls_score = self.box_predictor(box_features)
        #print(f"frcnn_cls_loc.shape:{frcnn_cls_loc.shape}, frcnn_cls_score.shape: {frcnn_cls_score.shape}")
        # (256，84) (256,21)

        result: List[Dict[str, torch.Tensor]] =[]
        frcnn_loss = torch.zeros([1])
        if self.training:
            if target_labels is None:
                raise ValueError("target labels cannot be None")
            if target_regressions is None:
                raise ValueError("target regressions cannot be None")
            frcnn_loss = self.compute_loss(frcnn_cls_score, target_labels, frcnn_cls_loc, target_regressions)
        else:
            boxes, scores, labels = self.postprocess_detections(frcnn_cls_score,
                                                                frcnn_cls_loc,
                                                                roi_proposals,
                                                                image_shape)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, frcnn_loss

