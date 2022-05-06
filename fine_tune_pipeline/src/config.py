import torch

BATCH_SIZE=8
RESIZE_TO = 800  #416
NUM_EPOCHS = 10
NUM_WORKERS =4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# data directory
TRAIN_DIR = '../data/Uno Cards.v2-raw.voc/train'
VALID_DIR = '../data/Uno Cards.v2-raw.voc/valid'

# 可预测的类别名，顺序和具体数据集相关
CLASSES = [
    '__background__', '11', '9', '13', '10', '6', '7', '0', '5', '4', '2', '14',
    '8', '12', '1', '3'
]

NUM_CLASSES = len(CLASSES)

# 训练前看下输入图片错没错，可选
VISUALIZE_TRANSFORMED_IMAGE = True

# 输出路径
OUTPUT_DIR = '../outputs'
