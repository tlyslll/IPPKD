import random

import torch
import torch.utils
import torch.utils.data.distributed
import numpy as np
import torchvision
from torchvision import datasets, transforms

def load_data(args):

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # load training data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    if args.is_split:
        class_data={i: [] for i in range(10)}

        for i,(_,target) in enumerate(trainset):
            class_data[target].append(i)
        #
        new_trainset=[]
        for labels in class_data.values():
            num_sample=int(len(labels)*args.split_rate)
            selected_idx=np.random.choice(labels,num_sample,replace=False)
            new_trainset.extend(selected_idx)

        new_trainset=torch.utils.data.Subset(trainset,new_trainset)

        train_loader = torch.utils.data.DataLoader(new_trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

