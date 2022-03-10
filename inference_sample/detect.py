import torchvision
import torch
import argparse
import cv2
import detect_utils

from PIL import Image

#建立argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help ='path to input image/video')
parser.add_argument('-m', '--min-size',dest='min_size',default=800,
                    help='minimum input size for the fasterRCNN network')
parser.add_argument('-t', '--threshold',default=0.8,
                    help='prediction score threshold for output boxes')
args = vars(parser.parse_args())

#下载预先训练好的fasterrcnn模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                             min_size=args['min_size'])
#设置训练硬件
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载图片
#image = Image.open(args['input'])#用rgb图片推测
image = cv2.imread(args['input'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model.eval().to(device)#把model放到预测模式下，很重要

boxes,classes,labels = detect_utils.predict(image,model,device,args['threshold'])
image = detect_utils.draw_boxes(boxes,classes,labels,image)
cv2.imshow('Output',image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg",image)
cv2.waitKey(0)