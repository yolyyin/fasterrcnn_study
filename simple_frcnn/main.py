import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from faster_rcnn import FasterRCNN


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


def grid_anchors(grid_width, grid_height, stride, anchor_base):
    # 计算原图所有anchor box的四角坐标
    grid_width, grid_height = (800 // 16, 800 // 16)
    stride = 16
    # 计算原图网格中心点
    shifts_x = torch.arange(0.5, grid_width + 0.5, dtype=torch.float32) * stride
    shifts_y = torch.arange(0.5, grid_height + 0.5, dtype=torch.float32) * stride
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    # flatten
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = torch.stack((shifts_y, shifts_x, shifts_y, shifts_x), dim=1)
    # 排列组合broadcasting
    anchors = (shifts.view(-1, 1, 4) + torch.as_tensor(anchor_base).view(1, -1, 4)).reshape(-1, 4)
    # print(anchors.shape)  # torch (22500,4)
    return anchors


def encode_boxes(anchors, bboxes):  # input (N,4) (M,4)
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
    dh = np.log(base_height / a_height)
    dw = np.log(base_width / a_width)
    encoded_locs = torch.vstack((dy, dx, dh, dw)).transpose(0, 1)
    return encoded_locs

def decode_boxes(anchors: torch.Tensor, roi_locs: torch.Tensor) -> torch.Tensor:
    #多张图片维度大于2的情况
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

    return pred_boxes  #(128,84)


def assign_targets_to_anchors(anchors, target_bboxes):
    # 设置 ahchor box对应的label和bbox loc
    # 暂时去除超过边框的anchor box
    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
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
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # [2262，2269...]

    # np.where的多维度使用比较复杂，以下是示例
    # a=np.array([[1,2],[3,4],[5,6],[3,6]])
    # b=np.array([3,4])
    # print(np.where(a==b))

    # 计算anchor_box对应的最大ious
    anchor_argmax_ious = ious.argmax(axis=1)
    anchor_max_ious = ious[np.arange(ious.shape[0]), anchor_argmax_ious]  # [0.06811669 0.07083762, ..]

    label = np.empty((ious.shape[0],), dtype=np.int32)
    label.fill(-1)
    pos_iou_threshold = 0.7
    neg_iou_threshold = 0.3
    pos_ratio = 0.5  # 最大positive label比例
    n_sample = 256  # 最多激活这么多anchor box
    label[anchor_max_ious < neg_iou_threshold] = 0  # 小于neg阈值的anchor
    label[gt_argmax_ious] = 1  # 最大iou anchor
    label[anchor_max_ious >= pos_iou_threshold] = 1  # 大于pos阈值的anchor
    # 调整一下label的正负标签比例
    # Faster_R-CNN paper: Each mini-batch arises from a single image that contains many positive and negative example anchors,
    # but this will bias towards negative samples as they are dominating.
    # Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch,
    # where the sampled positive and negative anchors have a ratio of up to 1:1.
    # If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones..
    n_pos = pos_ratio * n_sample
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1
    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1

    # 设置锚固框对应的bbox位置，格式为(y_center,x_center,log(h),log(w)) !确定一下这个格式
    max_iou_bbox = target_bboxes[anchor_argmax_ious]  # 每个valid anchor对应的bbox

    anchor_locs = encode_boxes(valid_anchor_boxes, max_iou_bbox)
    #print(anchor_locs.shape)  # (8940, 4)

    # 输出最终包括无效框的anchor box信息
    anchor_labels = torch.empty((len(anchors),), dtype=torch.int32)
    anchor_labels.fill_(-1)
    anchor_labels[index_inside] = torch.as_tensor(label, dtype = torch.int32)
    anchor_locations = torch.empty((len(anchors), 4), dtype=anchor_locs.dtype)
    anchor_locations.fill_(0.0)
    anchor_locations[index_inside, :] = anchor_locs
    #print(anchor_locations.shape, anchor_labels.shape)  # (22500, 4) (22500,)
    return anchor_locations, anchor_labels

def assign_targets_to_propos(rois,gt_bboxes,gt_labels,num_sample,num_pos,pos_iou_thresh,neg_iou_thresh_hi,neg_iou_tresh_lo):
    """
    随机选出一些roi pos(有物体框）,neg（背景） sample，其余的roi全部擦去
    输出： list(tensor), list(tensor), list(tensor)
    """
    target_regressions = []
    target_labels = []
    roi_samples =[]
    for rois_per_image, bboxes_per_image, labels_per_image in zip(rois, gt_bboxes, gt_labels):
        ious = torchvision.ops.box_iou(rois_per_image, bboxes_per_image)

        gt_assignment = ious.argmax(axis=1)  # 每个roi对应的bbox序号
        max_iou, _ = ious.max(axis=1)
        gt_roi_label = labels_per_image[gt_assignment]  # 每个roi对应的label

        pos_index = torch.where(max_iou >= pos_iou_thresh)[0]
        pos_this_image = int(min(num_pos, pos_index.shape[0]))
        if pos_this_image > 0:
            weights = torch.ones(pos_index.shape[0])
            rand_choice = weights.multinomial(pos_this_image, replacement = False)
            pos_index = pos_index[rand_choice]

        neg_index = torch.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_tresh_lo))[0]
        neg_this_image = int(min(num_sample - pos_this_image, neg_index.shape[0]))
        if neg_this_image>0:
            weights = torch.ones(neg_index.shape[0])
            rand_choice = weights.multinomial(neg_this_image, replacement=False)
            neg_index = neg_index[rand_choice]
        keep_index = torch.cat((pos_index, neg_index))
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_this_image:] = 0
        sample_rois = rois_per_image[keep_index, :]

        bboxes_for_sampled_roi = bboxes_per_image[gt_assignment[keep_index]]
        bboxes_attach_roi = encode_boxes(sample_rois, bboxes_for_sampled_roi)

        target_regressions.append(bboxes_attach_roi)
        target_labels.append(gt_roi_label)
        roi_samples.append(sample_rois)
    return target_labels, target_regressions,roi_samples


def filter_rpn_proposals(pred_rois, pred_object_scores, min_roi_size, n_pre_nms, n_post_nms, nms_thresh):
    """
    过滤rpn产生的roi proposal
    1.过滤掉面积太小的
    2.non max suppression
    """
    final_boxes = []
    final_scores = []
    for rois, scores in zip(pred_rois, pred_object_scores):
        # 暂时所有图片先都视为统一大小, clip roi到图片边缘
        rois[:, 0:4:2] = torch.clamp(rois[:, 0:4:2], 0, IMAGE_SIZE[0])
        rois[:, 1:4:2] = torch.clamp(rois[:, 1:4:2], 0, IMAGE_SIZE[1])
        # 过滤掉太小的roi
        hs = rois[:, 2] - rois[:, 0]
        ws = rois[:, 3] - rois[:, 1]
        keep = torch.where((hs >= min_roi_size) & (ws >= min_roi_size))[0]  # 还是对tensor用torch.where吧
        rois = rois[keep, :]
        scores = scores[keep]
        # 只保留n_pre_nms个rois
        order = scores.argsort(descending=True)[:n_pre_nms]
        rois = rois[order, :]
        scores = scores[order]
        # nms, 只保留n_post_nms个rois
        keep = torchvision.ops.nms(rois, scores, nms_thresh)
        keep = keep[:n_post_nms]
        final_scores.append(scores[keep])
        final_boxes.append(rois[keep,:])
    return final_boxes, final_scores

#############################
# 假装有个图片来示例
images = torch.zeros((1, 3, 800, 800)).float()
NUM_IMAGES = images.shape[0]
IMAGE_SIZE = (800, 800)
NUM_CLS = 21 #物体分成21类

gt_bboxes = torch.FloatTensor([[[20, 30, 400, 500], [300, 400, 500, 600]],
                               [[20, 30, 300, 500], [200, 400, 300, 600]]])  # 第一个图片中有两个框
#print(bboxes.shape)  # (2, 2, 4)
gt_labels = torch.LongTensor([[6, 8], [6, 3]])  #  (2,2)
sub_sample = 16  # 缩小16倍

# backbone -> feature map
# 使用vgg16,生成一个缩小了16倍的feature map
backbone_pre = torchvision.models.vgg16(pretrained=True)  # ！此处训练好之后可以换成自己的vgg16试一试
layer_list = list(backbone_pre.features.children())
layer_needed = []
temp = images.clone()

for l in layer_list:
    temp = l(temp)
    if temp.size()[2] < 800 // sub_sample:
        break
    layer_needed.append(l)

backbone = nn.Sequential(*layer_needed)

features = backbone(images).detach()  # (1, 512, 50, 50)
#!!!!  此处需要输出backbone的output channels!
backbone_out_channel = features.shape[1]

##########规范的target格式#############
targets = [{"boxes": torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]),
            "labels": torch.LongTensor([6, 8])},
           {"boxes": torch.FloatTensor([[20, 30, 300, 500], [200, 400, 300, 600]]),
            "labels": torch.LongTensor([6, 3])}]
image1 = torch.rand(3, 800, 800).float()
image2 = torch.rand(3, 800, 800).float()
image_list = [image1, image2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images = [img.to(device) for img in image_list]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

faster_rcnn = FasterRCNN(backbone, backbone_out_channel, NUM_CLS, (800, 800)).to(device)

faster_rcnn.eval()
detections = faster_rcnn(images)
faster_rcnn.train()
losses = faster_rcnn(images, targets)
print(f"losses:{losses}, losses.device:{losses['rpn_loss'].device}")










"""
#######新代码#####
fg_iou_thresh = 0.7
bg_iou_thresh = 0.3
n_loss_sample = 256  # 计算loss时对anchors的采样数
positive_fraction = 0.5  # 最大positive label比例
n_pre_nms = {"training": 12000, "testing": 6000}
n_post_nms = {"training": 2000, "testing": 300}
nms_thresh = 0.7
roi_min_size = 16

from anchor_utils import AnchorGenerator
from rpn import RPNHead, RegionProposalNetwork
anchor_generator = AnchorGenerator((8., 16., 32.), (0.5, 1., 2.))
rpn_head = RPNHead(features.shape[1], 9)
rpn = RegionProposalNetwork(anchor_generator,
                            rpn_head,
                            fg_iou_thresh,
                            bg_iou_thresh,
                            n_loss_sample,
                            positive_fraction,
                            n_pre_nms,
                            n_post_nms,
                            nms_thresh,
                            roi_min_size)
rpn.train()
rois, rpn_loss = rpn(images, features, targets)
print(f"roi[0] shape: {rois[0].shape}, rpn_loss: {rpn_loss}")

from roi_head import ROI_pool, FRCNN_classifier, RoIHeads
fg_iou_thresh = 0.5
bg_iou_thresh_hi = 0.5
bg_iou_thresh_lo = 0.0
n_loss_sample = 128
positive_fraction = 0.25
score_thresh = 0.05
nms_thresh = 0.5
detections_per_img = 100
roi_pool = ROI_pool((7, 7))
predictor = FRCNN_classifier(backbone_out_channel*7**2,NUM_CLS)
roi_head = RoIHeads(roi_pool,
                    predictor,
                    fg_iou_thresh,
                    bg_iou_thresh_hi,
                    bg_iou_thresh_lo,
                    n_loss_sample,
                    positive_fraction,
                    score_thresh,
                    nms_thresh,
                    detections_per_img)
roi_head.eval()
result, frcnn_loss = roi_head(features, rois, IMAGE_SIZE, targets)
print(f"result[0]: {result[0]}")
print(f"frcnn loss: {frcnn_loss}")
###################
"""


"""
#######anchor generator######

# generate anchors的常数
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]
NUM_ANCHORS = 9

##############    1. RPN target
# 把原图划分为feature map网格份，以每份中心点为中心画k个anchorbox
# feature map的每个像素点代表k倍的信息

# generate anchor box
hw_pairs = [(sub_sample * a_s * np.sqrt(r), sub_sample * a_s * np.sqrt(1. / r))
            for r in ratios for a_s in anchor_scales]
hw_pairs = np.asarray(hw_pairs, dtype=np.float32)
anchor_base = np.hstack((hw_pairs * (-1), hw_pairs)) / 2
# print(anchor_base)

anchors = grid_anchors(800, 800, sub_sample, anchor_base)  # torch.Size([22500, 4])


gt_roi_locs_list = []
gt_roi_labels_list = []
for i in range(NUM_IMAGES):
    target_bboxes = gt_bboxes[i]
    anchor_locations, anchor_labels = assign_targets_to_anchors(anchors, target_bboxes) #(22500,4) List(Tensor)
    gt_roi_locs_list.append(anchor_locations.unsqueeze(0))
    gt_roi_labels_list.append(anchor_labels.unsqueeze(0))

gt_roi_locs = torch.cat(gt_roi_locs_list, 0)#.view(-1, 4)
gt_roi_labels = torch.cat(gt_roi_labels_list, 0)#.view(-1)
print(gt_roi_labels.shape, gt_roi_locs.shape)  #(2,22500) (2,22500,4)


##############    2. RPN 推断
# fetures输入RPNHead,输出pred_obj_scores和pred_roi_locs
rpn_head = RPNHead(512, NUM_ANCHORS)
pred_obj_scores, pred_roi_locs = rpn_head(features)  # (1,9,50,50) (1,36,50,50)
#flatten成(N,-1,C)和(N,-1,4)样式
def permute_and_flatten(layer: torch.Tensor, N: int, C: int, H: int, W: int) -> torch.Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer
N, AxC, H, W = pred_obj_scores.shape
Ax4 = pred_roi_locs.shape[1]
A = Ax4//4
C = AxC//A
pred_obj_scores = permute_and_flatten(pred_obj_scores, N, C, H, W)  # (2,22500,1)
pred_roi_locs = permute_and_flatten(pred_roi_locs, N, 4, H, W)  # (2,22500,4)
print(pred_obj_scores.shape, pred_roi_locs.shape)

#decode boxes, 提出roi在原图的proposal, proposal的roi位置不再计入计算了，此处detach
pred_roi_locs = pred_roi_locs.detach()  # tensor (1,22500,4)
pred_object_scores = pred_obj_scores.flatten(-2).detach()  # tensor (1,22500)
rois = decode_boxes(anchors, pred_roi_locs) # tensor (1,22500,4)
#print(rois.shape)

##############    3. NMS 过滤ROI
# 过滤常数
nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_roi_size = 16
final_rois, final_scores = filter_rpn_proposals(rois, pred_object_scores, min_roi_size,
                                                n_train_pre_nms, n_train_post_nms, nms_thresh)
#print(final_rois, final_scores) #list(tensor) list(tensor)



##############    4. FASTER RCNN target
n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

n_pos = int(n_sample*pos_ratio)
target_rcnn_labels, target_rcnn_regs, roi_samples = assign_targets_to_propos(final_rois, gt_bboxes, gt_labels,n_sample,n_pos,
                                                 pos_iou_thresh, neg_iou_thresh_hi, neg_iou_thresh_lo)
target_rcnn_labels  = torch.cat(target_rcnn_labels, dim=0)
target_rcnn_regs  = torch.cat(target_rcnn_regs, dim=0)  #(256, 4)
print(target_rcnn_labels.shape, target_rcnn_regs.shape)   #（256，1） （256，4）
#print(roi_samples[0].shape)


##############    5. FASTER RCNN 推断
def convert_to_roi_format(boxes: [torch.Tensor]) -> torch.Tensor:
    concat_rois = torch.cat(boxes, dim=0)
    device, dtype = concat_rois.device, concat_rois.dtype
    ids = torch.cat([
        torch.full_like(b[:, :1], i, dtype=dtype, device=device)
        for i, b in enumerate(boxes)
    ], dim=0)
    concat_rois = torch.cat([ids, concat_rois], dim=1)
    return concat_rois
# roi_samples变形为带有图片ids的（N，5）形状
# 这个地方传入的roi是List[Tensor],后面postprocess_detection有用
concat_rois = convert_to_roi_format(roi_samples)
#print(concat_rois.shape) #(256，5)

# roi pool
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


size = (7, 7)
roi_pool = ROI_pool(size)
roi_output = roi_pool(features, concat_rois).flatten(start_dim=1)
#print(roi_output.shape)

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
        self.score_classifier = nn.Linear(4096,self.n_classes)
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


classifier_input_channel = roi_output.shape[-1]
classifier = FRCNN_classifier(classifier_input_channel, NUM_CLS)
frcnn_cls_loc, frcnn_cls_score = classifier(roi_output)
print(frcnn_cls_loc.shape, frcnn_cls_score.shape)  #(128，84) (128,21)

##############    7. 推测结果后处理
def postprocess_detection(class_logits,  # type Tensor (256,21)
                          box_regression,  # type Tensor (256,84)
                          roi_proposals,   # type list[Tensor]
                          image_shape,   #type Tuple[int,int]
                          score_thresh = 0.05,
                          detect_per_image = 100,
                          nms_thresh = 0.5
                          ): #() -> List[Dict(boxes:,scores:,labels:,)]
    device = class_logits.device
    n_classes = class_logits.shape[-1]

    #改变一下regression和logits的数据形式
    pred_boxes_encoded = box_regression  #(256,84)
    pred_scores = F.softmax(class_logits, -1)

    #分开之前concat的boxes到各个图片
    boxes_per_image = [roi_in_image.shape[0] for roi_in_image in roi_proposals]
    boxes_enconded_list = pred_boxes_encoded.split(boxes_per_image, 0)
    scores_list = pred_scores.split(boxes_per_image, 0)

    output = []

    #decode_boxes(anchors: torch.Tensor, roi_locs: torch.Tensor) 单张图片上
    for boxes_encoded, scores, roi in zip(boxes_enconded_list,scores_list,roi_proposals):
        boxes = decode_boxes(roi, boxes_encoded)  #(128, 84)
        boxes = boxes.view(-1, n_classes, 4)  # (128,21,4)
        boxes = ops.clip_boxes_to_image(boxes.view(-1, 4), image_shape).view(-1, n_classes, 4) #（128,21,4）

        #scores (128,21)
        #创建label Tensor
        labels = torch.arange(n_classes, device=device) #(1,21) 0到20
        labels = labels.view(1, -1).expand_as(scores) #(128,21)

        #去除背景预测
        boxes = boxes[:, 1:, :] #(128,20,4)
        scores = scores[:, 1:]  #(128,20)
        labels = labels[:, 1:]  #(128,20)

        #重新concat batch一下，因为有label在所以可以分辨哪个box是哪个物体类
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        #去除低分框
        inds = torch.where(scores > score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        #去除空框
        keep = ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        #nms suppression, 每个类独立地做，由batched_nms实现
        keep = ops.batched_nms(boxes, scores, labels, nms_thresh)
        #最多保留detection_per_image个
        keep = keep[:detect_per_image]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        output_per_image = dict(boxes=boxes, scores=scores, labels=labels)
        output.append(output_per_image)

    return output


#pred_output = postprocess_detection(frcnn_cls_score, frcnn_cls_loc, roi_samples, IMAGE_SIZE)
#print(len(pred_output))
#print(pred_output[0])



##############    6. 计算损失

###首先计算rpn loss
gt_roi_labels = gt_roi_labels.unsqueeze(-1)
rpn_cls_loss = F.binary_cross_entropy_with_logits(pred_obj_scores, gt_roi_labels.float())

#只对gt确认是正label的bbox计算边框损失
mask = (gt_roi_labels > 0).expand_as(pred_roi_locs)
mask_loc_preds = pred_roi_locs[mask].view(-1, 4)
mask_loc_targets = gt_roi_locs[mask].view(-1, 4)

def smooth(x):
    x = torch.abs(x)
    return ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))

def reg_loss(targets, preds, num_boxes):
    difference = targets - preds
    loss = smooth(difference).sum()/num_boxes
    return loss

num_loc = (gt_roi_labels > 0).sum()
rpn_loc_loss =reg_loss(mask_loc_targets, mask_loc_preds, num_loc)

rpn_lambda = 10.
sum_rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print(sum_rpn_loss)

###再来计算frcnn loss
frcnn_cls_loss = F.cross_entropy(frcnn_cls_score, target_rcnn_labels, ignore_index=-1)

#找来对的类对应的边框
n_sample = frcnn_cls_loc.shape[0]
frcnn_cls_loc = frcnn_cls_loc.view(n_sample, -1, 4)
#print(frcnn_cls_loss)
valid_loc = frcnn_cls_loc[torch.arange(0, n_sample), target_rcnn_labels]
#只对gt确认是正label的bbox计算边框损失
mask = (target_rcnn_labels > 0).unsqueeze(-1).expand_as(valid_loc)
valid_loc = valid_loc[mask].view(-1, 4)
target_loc = target_rcnn_regs[mask].view(-1, 4)

n_box = valid_loc.shape[0]
frcnn_loc_loss = reg_loss(target_loc, valid_loc, n_box)
#print(frcnn_loc_loss)
rcnn_lambda = 10.
sum_frcnn_loss = frcnn_cls_loss + (rcnn_lambda * frcnn_loc_loss)
print(sum_frcnn_loss)
"""


