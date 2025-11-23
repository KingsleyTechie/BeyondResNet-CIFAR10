import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_path, batch_size, test_batch_size, num_workers=2):
    """
    Create data loaders for CIFAR-10 dataset
    """
    # Basic normalization for CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    
    # Transform for training and testing
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
