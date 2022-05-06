import numpy as np
import cv2
import torch
import glob as glob
import os
import time

from model import create_model
from config import NUM_CLASSES,DEVICE,CLASSES

#创建一些随机颜色
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = create_model(NUM_CLASSES)
checkpoint = torch.load('../outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

#设置参数和图片路径
DIR_TEST = '../data/Uno Cards.v2-raw.voc/test'
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

# 图片储存路径
out_path = '../inference_outputs/images'

#设置inference分数阈值
detection_threshold = 0.8

#计算一下计算帧数
frame_count = 0
total_fps =0

for i in range(len(test_images)):
    # 得到图片名
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    # BGR to RGB
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    # 把hwc重新排序为chw
    image = np.transpose(image,(2,0,1))
    image = torch.tensor(image, dtype=torch.float)
    # 增加batch维度
    image = torch.unsqueeze(image, 0)

    # 预测
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # 得到帧数
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1

    # 释放outputs到cpu
    outputs = [{k: v.to('cpu') for k,v in t.items()} for t in outputs]

    # 取出boxes和classes和labels
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # 根据分数筛选boxes
        boxes = boxes[scores > detection_threshold].astype(np.int32)
        # 代画图框，从算法图上取下
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        #画框写字
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0,0,255), 2)
            cv2.putText(orig_image, class_name,
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                        2, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(out_path,f"{image_name}.jpg"), orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")