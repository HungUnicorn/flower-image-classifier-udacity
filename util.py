from PIL import Image
from torchvision import transforms, datasets
import torch
from torch.utils.data import DataLoader
from model import get_img_model
import numpy as np

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64)
    validationloader = DataLoader(validation_data, batch_size=64)

    return trainloader, testloader, validationloader, train_data.class_to_idx


def save_model(path, model, class_to_idx, hidden_units, arch):
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': class_to_idx,
                'hidden_units': hidden_units,
                'arch': arch},
               f'{path}checkout.pth')


def load_model():
    checkpoint = torch.load('checkout.pth')

    model = get_img_model(checkpoint['hidden_units'], checkpoint['arch'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image):
    img_pil = Image.open(image)

    predict_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = predict_transforms(img_pil)

    return np.array(img_tensor)
