import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES, OUTPUT_DIR

# 来个绘图风格
plt.style.use('ggplot')

# 约定俗成平均数类
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

# 存储最好模型类
class SaveBestModel:
    def __init__(self, best_valid_loss = float('inf')):
        self.best_valid_loss = best_valid_loss

    def __call__(self, current_valid_loss,
                 epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{OUTPUT_DIR}/best_model.pth")


# dataset的collate_fn, 本来dataset的输出格式是[(imgage1,targets1)..],
# 改变成（[image1,image2..],[targets1,targets2..])

def collate_fn(batch):
    # batch的collate_fn,把一个batch的所有图片做成一个list,所有target做成一个list
    return tuple(zip(*batch))

# 图片增强
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params= A.BboxParams(
        format = 'pascal_voc',
        label_fields = ['labels'])
    )

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

# 为了训练输入没错看一看
def show_transformed_image(train_loader):
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                              (box[0],box[1]),
                              (box[2],box[3]),
                              (0,0,255),2)
                cv2.putText(sample, CLASSES[labels[box_num]],
                            (box[0],box[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0,0,255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# 实时储存模型
def save_model(epoch, model, optimizer):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{OUTPUT_DIR}/last_model.pth')

# 储存loss图
def save_loss_plot(OUTPUT_DIR, train_loss, val_loss):
    figure1, train_ax = plt.subplots()
    figure2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure1.savefig(f"{OUTPUT_DIR}/train_loss.png")
    figure2.savefig(f"{OUTPUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')