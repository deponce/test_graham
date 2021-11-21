'''
This class defines all of the models that we will be using for the
experiments that we have to run
'''
import torch
import torchvision.models as models
from .scatter_resnet import scatternet_cnn
from .scatternet import scatternet
from .cifar_model import Net
model_dict = {
    "alexnet": models.alexnet(),
    "scatternet": scatternet(),
    "scatter_resnet": scatternet_cnn(),
    "resnet18": models.resnet18(pretrained=True),
    "cifar_net": Net(3, input_norm="BN")
}
