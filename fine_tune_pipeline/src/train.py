import os.path

from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUTPUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGE, NUM_WORKERS,
)

from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)

import torch
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')


# 训练函数
def train(train_data_loader, model):
    print('Training...')
    global train_itr
    global train_loss_list
    global train_loss_rec, optimizer

    # 初始化tqdm进度条,！！试一试total有什么用
    prog_bar = tqdm(train_data_loader, total = len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        # forward propagation
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        # 看一下loss_dict里面有什么
        #print(f"loss_dict: {loss_dict}")

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.detach().cpu().item()

        # train_loss_list为画loss图做准备，tran_loss_rec记录batch平均loss
        train_loss_list.append(loss_value)
        train_loss_rec.send(loss_value)

        # back propagation
        losses.backward()
        optimizer.step()
        train_itr += 1

        #在进度条旁显示递减的loss_value
        prog_bar.set_description(desc = f"Loss: {loss_value:.4f}")

        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()


    return train_loss_list


# 验证函数
def validate(valid_data_loader, model):
    print('Validating...')
    global val_itr
    global val_loss_list
    global val_loss_rec, optimizer

    # 初始化tqdm进度条,!!试一试total有什么用
    prog_bar = tqdm(valid_data_loader, total = len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        # forward propagation
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]
        with torch.no_grad():  # 不需要记录derivative
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.detach().cpu().item()

        # val_loss_list为画loss图做准备，val_loss_rec记录batch平均loss
        val_loss_list.append(loss_value)
        val_loss_rec.send(loss_value)

        val_itr += 1

        #在进度条旁显示递减的loss_value
        prog_bar.set_description(desc = f"Loss: {loss_value:.4f}")

        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()

    return val_loss_list

if __name__ == '__main__':
    # 初始化模型和optimizer
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists('../outputs/best_model.pth'):
        checkpoint = torch.load('../outputs/best_model.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 初始化数据集和dataloader
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # 初始化记录类和list类
    train_loss_rec = Averager()
    val_loss_rec = Averager()
    train_itr = 1
    val_itr = 1
    train_loss_list = []
    val_loss_list = []

    # 初始化SaveBestModel类
    save_best_model = SaveBestModel()

    # 如果要显示变化后的图片，显示一下
    if VISUALIZE_TRANSFORMED_IMAGE:
        from custom_utils import show_transformed_image
        show_transformed_image(train_loader)

    #开始训练
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        train_loss_rec.reset()
        val_loss_rec.reset()

        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_rec.value:.3f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_rec.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch + 1}")

        save_best_model(val_loss_rec.value, epoch, model, optimizer)
        save_model(epoch, model, optimizer)
        save_loss_plot(OUTPUT_DIR, train_loss, val_loss)

        time.sleep(10)