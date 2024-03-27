import torch
import numpy as np
from .topk import top1
from andesPrune.utils.utils import average, model_device


def training_set(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
    return optimizer,scheduler

def cust_forward_fn(model, data):
    device = model_device(model)
    return model(data[0].to(device=device))


def cust_loss_fn(output, data):
    target1 = data[1].to(device=output.device)
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(output, target1)


def distil_loss_fn(output, label):
    label = label.to(device=output.device)
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    return criterion(output, label)


def cust_eval_fn(output, data):
    # output=model(data[0].to(device=model_device(model)))
    target1 = data[1].to(device=output.device)
    return top1(output, target1)


def cust_inference_fn(model, dataset):

    with torch.no_grad():
        loss, acc = [], []
        for data in dataset:
            loss.append(cust_loss_fn(cust_forward_fn(model, data), data).item())
            acc.append(cust_eval_fn(cust_forward_fn(model, data), data).item())
    return {"Loss": average(loss)}, {"Top1": average(acc)}
