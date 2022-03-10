import torch

BATCH_SIZE = 4  # 一个batch的样本数，需根据GPU内存大小更改此值
RESIZE_TO = 512  # 缩放训练图片到此大小
NUM_EPOCHS = 100  # 训练多少epochs

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')

# training data directory
TRAIN_DIR = '../Microcontroller Detection/train'
# validation data directory
VALID_DIR = '../Microcontroller Detection/test'

# 探测物体类标签
CLASSES = ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora']
NUM_CLASSES = 5

# 是否看一看变化后的图像
VISUALIZE_TRANSFORMED_IMAGES = True

#输出路径
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # 这么多epoch后存储loss plots
SAVE_MODEL_EPOCH = 2 # 这么多epoch后存储model
