import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES as classes


# 此类储存训练和测试时的总损失和平均损失
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    # batch的collate_fn,把一个batch的所有图片做成一个list,所有target做成一个list
    return tuple(zip(*batch))


# 定义训练时为图片增强的transform
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',  # 标注bounding box的四个值代表什么
        label_fields=['labels'],  # 传入代表预测类名的变量叫什么
    ))


# 定义验证时的transform,仅变化为tensor
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


# 显示经过增强之后train_loader里的图片
def show_transformed_image(train_loader):
    if len(train_loader) > 0:
        for i in range(1): #此处这个1比较迷，暂时不管
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1,2,0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
