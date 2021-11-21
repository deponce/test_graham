import argparse
import torch.optim as optim
import numpy as np
import json
from time import time
from opacus import PrivacyEngine
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR
from utils import ImageNetData, load_data, verify_check_points_folder, training_monitor, load_checkpoint
from Optim import SGD_AGC
from models.models import model_dict
from imagenet import *
from train_utils import TrainOneEpoch, ValOneEpoch
from dp_utils import ORDERS, get_privacy_spent, get_renyi_divergence, scatter_normalization
from kymatio.torch import Scattering2D
import warnings
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.stdout.write('train model on:'+str(device)+'\n')
sys.stdout.flush()

def main(model_name="scatternet",
         data_path='~/projects/rrg-xihe/dataset/imagenet12',batch_size=256, mini_batch_size=2,
         epochs=10, optim="SGD", momentum=0.9,weight_decay=0,
         lr=1, lr_decay_epoch=[], lr_decay_factor=1,
         noise_multiplier=1, max_grad_norm=1, SGD_AGC_clip = np.inf, max_epsilon=None,
         val_batch_size=64,
         DP=True,
         CheckPointPATH='./checkpoint/', checkpoint=None):
    if not DP:
        print('WARNING: Turn PrivacyEngine off, the model would be trained by the optimizer without gradient clip and noise')
    verify_check_points_folder(CheckPointPATH)
    # load the data
    training_params = {
        "shuffle": True,
        "num_workers": 2,
        "batch_size": mini_batch_size
    }

    val_params = {
        "shuffle": False,
        "num_workers": 2,
        "batch_size": val_batch_size
    }
    if batch_size%mini_batch_size != 0:
        warnings.warn('batch_size should be an integral multiple of mini_batch_size')
        return None
    sys.stdout.write('start loading\n')
    sys.stdout.flush()
    train_data, test_data, val_data = load_data(data_path)
    # create dataloaders
    train_data = torch.utils.data.DataLoader(train_data, **training_params)
    val_data = torch.utils.data.DataLoader(val_data, **val_params)
    # test_data = torch.utils.data.DataLoader(test_data, **params)
    sys.stdout.write('Done\n')
    sys.stdout.flush()
    model = model_dict[model_name]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    else:
        optimizer = SGD_AGC(
                # The optimizer needs all parameter names
                # to filter them by hand later
                named_params=model.named_parameters(),
                lr=lr,
                momentum=momentum,
                clipping=SGD_AGC_clip,
                weight_decay=weight_decay,
                #nesterov=config['nesterov']
            )
    scheduler = MultiStepLR(optimizer, milestones=lr_decay_epoch, gamma=lr_decay_factor)
    if DP:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=batch_size / len(train_data.dataset),
            alphas=ORDERS,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        ).to(device)

        # TODO: fix the privacy engine parameters
        privacy_engine.attach(optimizer)

    #monitor = training_monitor()
    start_epoch=0
    # load check point
    if checkpoint:
        model, optimizer, start_epoch, train_loss = load_checkpoint(checkpoint=CheckPointPATH+checkpoint,
                                                              model=model, optimizer=optimizer,
                                                              epoch=0, train_loss=0)
    epoch_train_loss = []
    epoch_train_acc_top1 = []
    epoch_train_acc_top5 = []
    epoch_val_loss = []
    epoch_val_top1_acc = []
    epoch_val_top5_acc = []
    epoch_epsilon = []
    # main training loop
    scattering = Scattering2D(J=3, shape=(224, 224)).to(device)

    #rdp_norm = 0

    for epoch in range(start_epoch+1, epochs+1):
        sys.stdout.write('EPOCH: '+str(epoch)+'\n')
        sys.stdout.flush()
        #monitor.reset()
        start = time()

        train_acc_top1, train_acc_top5, train_loss = TrainOneEpoch(model=model, scattering=scattering,
                                                                   criterion=criterion, train_data=train_data,
                                                                   optimizer=optimizer, device=device,
                                                                   steps=batch_size//mini_batch_size, DP=DP)

        val_top1_acc, val_top5_acc, val_loss =ValOneEpoch(model=model, scattering=scattering,
                                                          criterion=criterion, train_data=val_data,
                                                          device=device)
        #val_top1_acc, val_top5_acc, val_loss = monitor.get_acc_loss()
        scheduler.step()
        end = time()

        if DP:
            rdp_sgd = get_renyi_divergence(
                privacy_engine.sample_rate, privacy_engine.noise_multiplier
                )* privacy_engine.steps
            #epsilon, _ = get_privacy_spent(rdp_norm + rdp_sgd)
            epsilon2, _ = get_privacy_spent(rdp_sgd)
            #print(f"ε = {epsilon2:.3f} (sgd only: ε = {epsilon2:.3f})")
            print(f"ε = {epsilon2:.3f} ")
            if max_epsilon is not None and epsilon >= max_epsilon:
                return
            else:
                epsilon = None

        sys.stdout.write('saving\n')
        sys.stdout.flush()

        epoch_train_loss.append(train_loss)
        epoch_train_acc_top1.append(train_acc_top1.item())
        epoch_train_acc_top5.append(train_acc_top5.item())
        epoch_val_loss.append(val_loss)
        epoch_val_top1_acc.append(val_top1_acc.item())
        epoch_val_top5_acc.append(val_top5_acc.item())

        CheckPoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'train_loss':epoch_train_loss,
                      'train_acc_top1':epoch_train_acc_top1,
                      'train_acc_top5':epoch_train_acc_top5,
                      'val_loss': epoch_val_loss,
                      'val_acc_top1': epoch_val_top1_acc,
                      'val_acc_top5': epoch_val_top5_acc,
                      }
        if DP:
            epoch_epsilon.append(epsilon2)
            CheckPoint['epsilon']=epoch_epsilon

        #print('| val_loss: ',val_loss,'| val_top1_acc: ',val_top1_acc.item(),
        #      '| val_top5_acc: ',val_top5_acc.item(),'|')


        print('| epoch: ',epoch, '| time: ', end-start,'| train_loss: ',train_loss,
              '| train_acc_top1: ',train_acc_top1.item(),'| train_acc_top5: ',train_acc_top5.item(),
              '| val_loss: ',val_loss,'| val_top1_acc: ',val_top1_acc.item(),
              '| val_top5_acc: ',val_top5_acc.item(),'|')

        sys.stdout.flush()
        torch.save(CheckPoint, CheckPointPATH + model_name + 'epoch' + str(epoch) + '.pt')

    epoch_train_loss = np.array(epoch_train_loss)
    epoch_train_acc_top1 = np.array(epoch_train_acc_top1)
    epoch_train_acc_top5 = np.array(epoch_train_acc_top5)
    epoch_val_loss = np.array(epoch_val_loss)
    epoch_val_top1_acc = np.array(epoch_val_top1_acc)
    epoch_val_top5_acc = np.array(epoch_val_top5_acc)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_train_loss',epoch_train_loss)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_train_acc_top1',epoch_train_acc_top1)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_train_acc_top5',epoch_train_acc_top5)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_val_loss',epoch_val_loss)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_val_top1_acc',epoch_val_top1_acc)
    np.save(CheckPointPATH+'acc_loss/'+'epoch_val_top5_acc',epoch_val_top5_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="cifar_net", choices=["alexnet", "scatternet", "scatter_resnet", "resnet18", "cifar_net"])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--mini_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default="SGD", choices=["SGD", "Adam", 'SGD_AGC'])
    parser.add_argument('--SGD_AGC_clip', type=float, default=999999)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('-lr_decay_epoch', nargs='+', help='<Required> Set flag', required=False, default=[])
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)

    parser.add_argument('--DP', dest='DP', action='store_true')
    parser.add_argument('--No-DP', dest='DP', action='store_false')
    parser.set_defaults(DP=False)

    parser.add_argument('--noise_multiplier', type=float, default=1.3)
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--max_epsilon', type=float, default=None)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_path',type=str,default='./data')
    parser.add_argument('--CheckPointPATH', type=str, default='CheckPoint/')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
