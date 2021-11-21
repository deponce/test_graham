'''
Dataloader for imagenet
'''

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from imagenet import *
import torch.nn as nn
import os
import warnings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class training_monitor():
    def __init__(self):
        self._top1cnt = 0
        self._top5cnt = 0
        self._totalcnt = 0
        self._training_loss = 0
    def reset(self):
        self._top1cnt = 0
        self._top5cnt = 0
        self._totalcnt = 0
        self._training_loss = 0
    def update(self,model_output, target, loss):
        with torch.no_grad():
            self._totalcnt += len(model_output)
            self._training_loss += loss.item()
            top_5 = torch.topk(model_output,dim=1,k=5,
                               largest=True, sorted=True).indices
            top_5_diff = torch.abs(top_5-target[:,None])<=0.1
            self._top5cnt += torch.sum(top_5_diff)
            top_1_diff = torch.abs(top_5[:,0]-target)<=0.1
            self._top1cnt += torch.sum(top_1_diff)
    def avg_loss(self):
        return self._training_loss/self._totalcnt
    def top1accuracy(self):
        return self._top1cnt/self._totalcnt
    def top5accuracy(self):
        return self._top5cnt/self._totalcnt
    def get_acc_loss(self):
        return [self.top1accuracy(),self.top5accuracy(),self.avg_loss()]

class ImageNetData(Dataset):
    def __init__(self, samples=None, targets=None, transform=None, augment=False, height=256, width=256, C_in=3):
        self.samples = samples
        self.targets = targets
        self.transform = transform
        self.augment = augment
        self.height = height
        self.width = width
        self.C_in = C_in
    def __len__(self):
        return self.samples.shape[0]
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform is not None:
            # should be a single sample at a time here
            sample = sample.reshape(self.height, self.width, self.C_in)
            sample = self.transform(sample)
        # recasting
        sample = np.array(sample).astype(np.double)
        sample = torch.from_numpy(sample)
        # if we're augmenting the data, then the PIL images
        # will return the domain of the image to [0,255]
        # so we need to renormalize each sample
        if self.augment:
            return (sample / 255.) - 0.5, self.targets[idx]
        # on the other hand, if we are not augmenting, there is no need for this additional normalization
        return sample, self.targets[idx]


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_data(filepath= './data', augment=True):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = [transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize]
    if augment:
        train_transforms = [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize]
    """
    train_set = datasets.ImageNet(root=filepath, split='val',
                                  transform=transforms.Compose(train_transforms))

    test_set = datasets.ImageNet(root=filepath, split='val',
                                 transform=transforms.Compose(
                                     [transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(), normalize]))
    
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    test_set = datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)

    return train_set, None,  test_set

def verify_check_points_folder(path='./checkpoint/'):
    if os.path.isdir(path):
        print('check points folder is exist')
        if os.listdir(path):
            warnings.warn('check points folder is not empty')
    else:
        print('creat a check points folder:', path)
        os.makedirs(path)

    if not os.path.isdir(path+'acc_loss/'):
        os.makedirs(path+'acc_loss/')
    return None

def load_checkpoint(checkpoint, model, optimizer, epoch, train_loss,device='cuda'):
    try:
        if device == 'cuda':
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
    except:
        warnings.warn('the checkpoint file do not exist')
    return model, optimizer, epoch, train_loss
