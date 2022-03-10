import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset,DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

# the dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        #得到所有图片路径，得到所有图片名并排序
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # 得到图片名和路径
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        image = cv2.imread(image_path)
        # 把opencv BGR格式转为RGB格式
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # 得到target文件名和路径
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        #解析xml文件树，得到boxes和labels
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        #得到旧图片的高宽
        image_height = image.shape[0]
        image_width = image.shape[1]

        for member in root.findall('object'):
            # 通过xml中的物体名找到classes列表中的物体类别编号，添加到label列表
            labels.append(self.classes.index(member.find('name').text))

            #得到左上点和右下点坐标
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            #resize bounding box
            xmin = (xmin/image_width)*self.width
            ymin = (ymin/image_height)*self.height
            xmax = (xmax/image_width)*self.width
            ymax = (ymax/image_height)*self.height

            boxes.append([xmin, ymin, xmax, ymax])

        n_boxes = np.array(boxes)
        n_labels = np.array(labels)

        # 转化为tensor
        boxes = torch.as_tensor(n_boxes,dtype=torch.float32)
        labels = torch.as_tensor(n_labels,dtype=torch.int64)
        # box面积
        areas = (boxes[:,3]-boxes[:,1]) * (boxes[:,2]-boxes[:,0])
        # no crowd instance, 不懂
        #iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # 集合到字典target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['areas'] = areas
        #target['iscroud'] = iscrowd
        image_id = torch.tensor(np.array([idx]))
        target['image_id'] = image_id

        #应用图片转换
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = target['labels'])
            image_resized = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
train_dataset = MicrocontrollerDataset(TRAIN_DIR,RESIZE_TO,RESIZE_TO,CLASSES,get_train_transform())
valid_dataset = MicrocontrollerDataset(VALID_DIR,RESIZE_TO,RESIZE_TO,CLASSES,get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

#执行datasets.py来可视化一些样本图片
if __name__ == '__main__':
    dataset = MicrocontrollerDataset(
        TRAIN_DIR,RESIZE_TO,RESIZE_TO,CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    #可视化一张图片
    def visualize_sample(image, target):
        for index, box in enumerate(target['boxes']):
            label = CLASSES[target['labels'][index]]
            cv2.rectangle(
                image,
                (int(box[0]),int(box[1])),
                (int(box[2]),int(box[3])),
                (0,255,0), 1
            )

            cv2.putText(
                image,label,(int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2
            )
        cv2.imshow('Image',image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)





