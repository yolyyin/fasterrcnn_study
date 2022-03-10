import torchvision.transforms as transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

#create different label color for coco categories
COLORS = np.random.uniform(0,255,size = (len(coco_names),3))

#define torch transformation from image to tensor
tensor_trans = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image,model,device,detection_threshold):
    #变化图片为tensor
    image = tensor_trans(image).to(device)
    image = image.unsqueeze(0) #增加batch维度
    outputs = model(image) #得到预测结果

    #分别打印预测结果
    print(f"BOXES: {outputs[0]['boxes']}")
    print(f"LABELS: {outputs[0]['labels']}")
    print(f"SCORES: {outputs[0]['scores']}")

    #得到所有预测结果
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    #得到超过阈值的boxes,用pred_scores做index
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    #此处没有滤去pred_classes和labels中小于阈值的项，是因为模型默认输出分数由大到小的项
    return boxes,pred_classes,outputs[0]['labels']#框坐标，类名，类label数

#结果画框
def draw_boxes(boxes,classes,labels,image):
    #opencv读取图片,opencv是bgr,pil是rgb
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)#RGB转BGR方便openCV存储
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]),int(box[1])),
            (int(box[2]),int(box[3])),
            color,2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image
