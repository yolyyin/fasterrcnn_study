from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

# 训练函数
def train(train_data_loader, model):
    print('Training...')
    global train_itr
    global train_loss_list
    #global train_loss_rec, optimizer

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
        optimizer.zero_grad()
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
    # 初始化模型
    model = create_model(num_classes = NUM_CLASSES)
    model = model.to(DEVICE)
    # 得到模型参数, 直接船model.parameters()也可以吧
    params = [p for p in model.parameters() if p.requires_grad]
    # 定义随机梯度下降optimizer, !!momentum和weight_decay再看看
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 初始化记录类和list类
    train_loss_rec = Averager()
    val_loss_rec = Averager()
    train_itr = 1
    val_itr = 1
    train_loss_list = []
    val_loss_list = []

    # pytorch存储模型名
    MODEL_NAME = 'model'

    #如果要显示变化后的图片，显示一下
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_transformed_image
        show_transformed_image(train_loader)

    #开始训练
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        train_loss_rec.reset()
        val_loss_rec.reset()

        #画两个图
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # 训练，记录时间
        start = time.time()
        train_loss = train(train_loader, model)
        #train_loss = [0,0]
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_rec.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_rec.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        if(epoch+1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')

        if (epoch+1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(train_loss, color = 'blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color = 'red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')

        if (epoch+1) == NUM_EPOCHS:  # 最后存一存
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch + 1}.png")

            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")

        plt.close('all')
