import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
from config import config

def get_transforms(img_size):
    train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                          transforms.RandomCrop(img_size, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) # dataset default normazation constants
    
    test_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    return train_transform, test_transform

def build_dataloaders(dataset_name="cifar10", subset_size=None, img_size=64):
    train_transform, test_transform = get_transforms(img_size)
    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    if subset_size is not None and subset_size < len(trainset):
        indices = list(range(len(trainset)))
        random.seed(config.random_seed)
        random.shuffle(indices)
        trainset = Subset(trainset, indices[:subset_size])

    trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader