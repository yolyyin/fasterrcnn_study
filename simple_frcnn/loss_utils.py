import torch


def smooth(x:torch.Tensor):
    x = torch.abs(x)
    return ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))


def reg_loss(targets:torch.Tensor, preds:torch.Tensor, num_boxes:int):
    loss = torch.zeros(1, device=targets.device)
    difference = targets - preds
    if num_boxes != 0:
        loss = smooth(difference).sum()/num_boxes
    return loss

