import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

class GeneralizedRCNNTransform(nn.Module):
    """
    对输入GeneralizedRCNN的数据进行预先transform处理

    transform包括：1.normalization 2.resize image到统一大小
    返回tensor格式的images, 和List[Dict[Tensor]]格式的targets
    """
    def __init__(self,
                 image_size: Tuple[int,int],
                 image_mean: List[float],
                 image_std: List[float],
                 ):
        super().__init__()
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std


    def forward(self,
                images: List[torch.Tensor],
                targets: Optional[List[Dict[str, torch.Tensor]]]
                ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        images = [img for img in images]  # 复制一遍
        if targets is not None:
            targets = [{k: v for k, v in t.items()} for t in targets]  # 复制一遍

        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None

            image = self.normalize(image)
            resized_image, resized_target = self.resize(image, target)
            images[i] = image.unsqueeze(0)
            if targets is not None and resized_target is not None:
                targets[i] = resized_target

        batched_images = torch.cat(images, dim=0)
        #print(f"batched_images.shape: {batched_images.shape}")
        return batched_images, targets


    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]


    def resize(
            self,
            image: torch.Tensor,
            target: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        h, w = image.shape[-2:]  # 缓存原图大小
        #缩小图片

        image = F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def resize_boxes(self,
                     boxes: torch.Tensor,
                     original_size: Tuple[int, int],
                     new_size: Tuple[int, int]) -> torch.Tensor:
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)  # dim1切片

        xmin = xmin * ratio_width   # 此处还是小心一点吧，图片是(N,H,W,C)，而pascal voc格式确实是(xmin,ymin,xmax,ymax)
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)


    def postprocess(
            self,
            result: List[Dict[str, torch.Tensor]],
            image_size: Tuple[int, int],
            original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        if self.training:
            return result  #反正返回的都是空[]
        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = self.resize_boxes(boxes, image_size, o_im_s)
            result[i]["boxes"] = boxes
        return result




class GeneralizedRCNN(nn.Module):
    """
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self,
                 backbone: nn.Module,
                 rpn: nn.Module,
                 roi_head: nn.Module,
                 transform: nn.Module):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        return detections

    def forward(self,
                images,  # type: List[Tensor]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):
        """

        Return:
            result (list[Dict] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[Dict] contains 'boxes', 'scores', 'labels'
        """
        #检查一下target格式
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        #检查image格式，储存image原大小
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        #transform
        batched_images, targets = self.transform(images, targets)

        # 检查target boxes格式正确
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = (boxes[:, 2:] <= boxes[:, :2])
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        #推导
        features = self.backbone(batched_images)
        #print(f"features.shape: {features.shape}")
        rois, rpn_loss = self.rpn(batched_images, features, targets)
        #print(f"rois[0].shape: {rois[0].shape}, rpn loss:{rpn_loss}")
        image_size = batched_images.shape[-2:]
        detections, frcnn_loss = self.roi_head(features, rois, image_size, targets)
        detections = self.transform.postprocess(detections, image_size, original_image_sizes)

        losses = dict(rpn_loss=rpn_loss, frcnn_loss=frcnn_loss)
        return self.eager_outputs(losses, detections)

