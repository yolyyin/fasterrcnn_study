import torchvision
import torch
import argparse
import time
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

#设置视频捕捉的输入输出object
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error trying to read video.')

#捕捉视频框的高宽
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
#初始化视频输出实例
out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),30,
                      (frame_width,frame_height))

frame_count = 0
total_fps = 0 #统计总帧数
model = model.eval().to(device)

#读取视频直到结束
while(cap.isOpened()):
    #捕捉输入视频的一帧
    ret, frame = cap.read()
    if ret == True:
        #得到开始时间
        start_time = time.time()
        with torch.no_grad():
            #得到预测结果
            boxes, classes, labels = detect_utils.predict(frame, model, device, args['threshold'])
        #画出结果帧
        image = detect_utils.draw_boxes(boxes,classes,labels,frame)

        #得到结束时间
        end_time = time.time()
        #算出帧数
        fps = 1 / (end_time - start_time)
        #把fps加到总fps
        total_fps += fps
        #增加frame count
        frame_count += 1

        #添加热键q结束视频采样
        wait_time = max(1, int(fps/4))
        cv2.imshow('Output', image)
        out.write(image)
        #此句检查key的unicode编码，0xFF是24个0和8个1
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

#释放视频捕捉器
cap.release()
cv2.destroyAllWindows()

#计算平均帧数
avg_fps = total_fps/frame_count
print(f"average FPS: {avg_fps:.3f}")