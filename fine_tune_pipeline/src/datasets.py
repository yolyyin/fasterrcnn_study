import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform


#自定义数据类
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        #得到所有图片路径
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # 获得图片名
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path,image_name)

        # 读取图片
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # 获得标注文件名
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels =[]
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        image_width = image.shape[1]
        image_height = image.shape[0]

        # 解析xml,存入boxes,labels
        for member in root.findall('object'):
            #解析物体标签
            name = member.find('name').text
            labels.append(self.classes.index(name))

            #解析bbox数据, 左上角和右下角
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            #缩放bbox点数据
            xmin = (xmin / image_width) * self.width
            xmax = (xmax / image_width) * self.width
            ymin = (ymin / image_height) * self.height
            ymax = (ymax / image_height) * self.height

            boxes.append([xmin, ymin, xmax, ymax])

        # 全部转换为tensor
        # bounding box to tensor
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        # area of the bounding boxes
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        # iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        # 存入target字典
        target = dict(boxes= boxes,
                      labels = labels,
                      areas = areas,
                      #iscrowd = iscrowd,
                      image_id = torch.tensor(np.array([idx])))

        # 应用图片增强
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes = target['boxes'],
                                     labels = target['labels'])
            image_resized = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return  len(self.all_images)


# 准备训练集和验证集
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR,RESIZE_TO,RESIZE_TO,CLASSES,get_train_transform())
    return train_dataset


def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader


def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader


# 检查一下数据集代码免得弄错了
if __name__ == '__main__':
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")


    # 查看单一图片和标注
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )

            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target= dataset[i]
        visualize_sample(image, target)

