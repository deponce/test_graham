
import time
import sys
import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm
def TrainOneEpoch(model, scattering, criterion, train_data,optimizer, device, steps=60,DP=True):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.train()
    for idx, data in enumerate(train_data, 0):
        batch, target = data
        start = time.time()
        batch, target = batch.to(device), target.to(device)
        #batch = scattering(batch)
        outputs = model(batch)
        loss = criterion(outputs, target)
        loss.backward()
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(acc1[0], batch.size(0))
        top5.update(acc5[0], batch.size(0))
        if (idx+1)%steps==0:
            optimizer.step()
            optimizer.zero_grad()
        elif DP:
            with torch.no_grad():
                optimizer.virtual_step()
        end = time.time()
        if idx%1000==0:
            sys.stdout.write(str(idx+1)+' batch take '+str(end-start)+'s\n')
            sys.stdout.flush()
        if idx>10:
            break
    optimizer.zero_grad()
    return top1.avg, top5.avg, losses.avg

def ValOneEpoch(model, scattering, criterion, train_data, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(train_data, 0)):
            batch, target = data
            batch, target = batch.to(device), target.to(device)
            #batch = scattering(batch)
            outputs = model(batch)
            loss = criterion(outputs, target)
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            losses.update(loss.item(), batch.size(0))
            top1.update(acc1[0], batch.size(0))
            top5.update(acc5[0], batch.size(0))
    return top1.avg, top5.avg, losses.avg

