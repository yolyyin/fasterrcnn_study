import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):

    # load faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # !!得到所需input features数，不懂再看看
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 定义一个有新的num_classes的roi_heads
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    return model

