import time
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models


PRETRAINED_MODELS = {
    'vgg11': models.vgg11,
    'densenet161': models.densenet161

}


def get_img_model(hidden_units, arch):
    model = PRETRAINED_MODELS[arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(0.2)),
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model
