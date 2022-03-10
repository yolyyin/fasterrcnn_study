import numpy as np
import cv2
import torch
import glob as glob

from model import create_model
from config import NUM_CLASSES, CLASSES

#配置设备和模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_model(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('../outputs/model20.pth', map_location= device))
model.eval()

#设置参数和图片路径
DIR_TEST = '../test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

#设置inference分数阈值
detection_threshold = 0.8

for i in range(len(test_images)):
    # 得到图片名
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    # BGR to RGB
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    # 把hwc重新排序为chw
    image = np.transpose(image,(2,0,1)) #.astype(np.float)
    image = torch.tensor(image, dtype=torch.float).cuda()
    # 增加batch维度
    image = torch.unsqueeze(image, 0)
    #预测
    with torch.no_grad():
        outputs = model(image)

    # 释放outputs到cpu
    outputs = [{k: v.to('cpu') for k,v in t.items()} for t in outputs]

    # 取出boxes和classes和labels
    if len(outputs[0]['boxes']) !=0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # 根据分数筛选boxes
        boxes = boxes[scores > detection_threshold].astype(np.int32)
        # 代画图框，从算法图上取下
        draw_boxes = boxes.copy()
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        #画框写字
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0,0,255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),
                        2, lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)
        cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()