import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def get_data(data_dir, train_dir=None, valid_dir=None, test_dir = None):
    if train_dir == None:
        train_dir = data_dir + '/train'
    if valid_dir == None:
        valid_dir = data_dir + '/valid'
    if test_dir == None:
        test_dir = data_dir + '/test'
    
    tensor_normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    crop_tens_norm = [transforms.CenterCrop(224)] + tensor_normalize
    batch_size = 50

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ] + tensor_normalize
        ),
        "test" : transforms.Compose(crop_tens_norm),
        "valid": transforms.Compose(crop_tens_norm),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "train": ImageFolder(train_dir, transform=data_transforms["train"]),
        "test" : ImageFolder(test_dir , transform=data_transforms["test"]),
        "valid": ImageFolder(valid_dir, transform=data_transforms["valid"])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True ),
        "test" : torch.utils.data.DataLoader(image_datasets["test"] , batch_size=batch_size ),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batch_size ),
    }
    
    return dataloaders, image_datasets, data_transforms