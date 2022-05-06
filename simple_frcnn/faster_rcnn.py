from generalized_rcnn import GeneralizedRCNN, GeneralizedRCNNTransform
from anchor_utils import AnchorGenerator
from rpn import RPNHead, RegionProposalNetwork
from roi_head import ROI_pool, FRCNN_classifier, RoIHeads


class FasterRCNN(GeneralizedRCNN):
    """
    Args:

    """
    def __init__(self,
                 backbone,
                 backbone_out_channel,
                 num_classes=None,
                 # transform paras
                 image_size=(800, 800),
                 image_mean=None,
                 image_std=None,
                 # rpn paras
                 rpn_anchor_generator=None,
                 rpn_head=None,
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_n_loss_sample=256,
                 rpn_positive_fraction=0.5,
                 rpn_nms_thresh=0.7,
                 rpn_n_pre_nms_train=12000,
                 rpn_n_pre_nms_test=6000,
                 rpn_n_post_nms_train=2000,
                 rpn_n_post_nms_test=300,
                 rpn_score_thresh=0.0,
                 rpn_min_size=16,
                 # box paras
                 box_roi_pool=None,
                 box_predictor=None,
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh_hi=0.5,
                 box_bg_iou_thresh_lo=0.0,
                 box_n_loss_sample=128,
                 box_positive_fraction=0.25,
                 box_socre_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100
                 ):

        #检查一下输入格式

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )

        if not isinstance(box_roi_pool, (ROI_pool, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type ROI_pool or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        #准备rpn部分

        if rpn_anchor_generator is None:
            rpn_anchor_generator = AnchorGenerator((8., 16., 32.), (0.5, 1., 2.))
        if rpn_head is None:
            rpn_head = RPNHead(backbone_out_channel, rpn_anchor_generator.num_anchors_per_location())

        rpn_n_pre_nms = dict(training=rpn_n_pre_nms_train, testing=rpn_n_pre_nms_test)
        rpn_n_post_nms = dict(training=rpn_n_post_nms_train, testing=rpn_n_post_nms_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_n_loss_sample,
            rpn_positive_fraction,
            rpn_n_pre_nms,
            rpn_n_post_nms,
            rpn_nms_thresh,
            rpn_min_size,
            rpn_score_thresh
        )

        #准备roi_head部分

        if box_roi_pool is None:
            box_roi_pool = ROI_pool((7, 7))

        if box_predictor is None:
            patch_size = box_roi_pool.out_size
            predictor_in_channels = backbone_out_channel*patch_size[0]*patch_size[1]
            box_predictor = FRCNN_classifier(predictor_in_channels, num_classes)

        roi_head = RoIHeads(
            box_roi_pool,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh_hi,
            box_bg_iou_thresh_lo,
            box_n_loss_sample,
            box_positive_fraction,
            box_socre_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        #准备输入数据transform类

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(image_size, image_mean, image_std)

        super().__init__(backbone, rpn, roi_head, transform)

