import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import random
import pandas as pd
import os

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_data_transforms(jitter_coefficient=0.15):
    """
    Get the data transformations for training, validation, and testing
    
    Args:
        jitter_coefficient (float): Coefficient for color jitter
        
    Returns:
        dict: Dictionary containing 'train' and 'val_test' transforms
    """
    
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # For our training data, we will apply data augmentation to improve generalization and help the model become more robust to different variations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)), # Crop at random position and resizes, set scale as 0.7 to 1.0 to avoid too small crops and lose the flower
        transforms.RandomHorizontalFlip(), # This randomly flips images horizontally
        transforms.RandomVerticalFlip(p=0.2),  # Add vertical flips (flowers viewed from any angle)
        transforms.RandomRotation(30), # Randomly rotates the image within +- 30 degrees
        transforms.ColorJitter(
            brightness=jitter_coefficient, 
            contrast=jitter_coefficient, 
            saturation=jitter_coefficient, 
            hue=jitter_coefficient)
        , # Color transformations
        transforms.ToTensor(), # Converts the image to a PyTorch tensor
        transforms.Normalize(std=normalize_std, mean=normalize_mean) # Normalizes the image with mean and std deviation (from ImageNet)
    ])

    # For validation and testing data, we should only resize and normalize to ensure consistent results
    val_test_transforms = transforms.Compose([
        transforms.Resize(256), # Resizes the image to 256x256 abit larger than needed
        transforms.CenterCrop(224), # Consistent center crop
        transforms.ToTensor(), # Converts the image to a PyTorch tensor
        transforms.Normalize(std=normalize_std, mean=normalize_mean) # Normalizes the image with mean and std deviation (from ImageNet)
    ])
    
    return {
        'train': train_transforms,
        'val_test': val_test_transforms
    }
    
    
def load_flowers102_dataset(data_dir='./data'):
    """
    Load the Flowers102 dataset
    
    Args:
        data_dir (str): Directory to store the dataset
        
    Returns:
        tuple: (datasets_dict, dataloaders_dict, dataset_sizes)
    """
    # Get data transforms
    transforms_dict = get_data_transforms()
    
    # Load datasets
    datasets_dict = {
        'train': datasets.Flowers102(
            root=data_dir, 
            split='train', 
            transform=transforms_dict['train'], 
            download=True
        ),
        'val': datasets.Flowers102(
            root=data_dir, 
            split='val', 
            transform=transforms_dict['val_test'], 
            download=True
        ),
        'test': datasets.Flowers102(
            root=data_dir, 
            split='test', 
            transform=transforms_dict['val_test'], 
            download=True
        )
    }
    
    # Print dataset sizes
    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val', 'test']}
    print(f"Train set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")
    print(f"Test set size: {dataset_sizes['test']}")
    
    return datasets_dict

def create_dataloaders(datasets_dict, batch_size=32, num_workers=4):
    """
    Create DataLoaders for each dataset split
    
    Args:
        datasets_dict (dict): Dictionary containing 'train', 'val', and 'test' datasets
        batch_size (int): Batch size for training and evaluation
        num_workers (int): Number of worker threads for loading data
        
    Returns:
        dict: Dictionary containing DataLoaders for each split
    """
    # Create DataLoaders
    dataloaders_dict = {
        'train': DataLoader(
            datasets_dict['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets_dict['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets_dict['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders_dict


def get_flower_labels(path):
    """
    Load flower labels from a text file.

    Args:
        path (str): Path to the file containing flower labels (e.g., 'flower_labels.txt').

    Returns:
        list: A list of flower labels as strings.
    """
    try:
        # Read the labels from the text file
        with open(path, 'r') as f:
            flower_labels = [line.strip().replace("'", "") for line in f.readlines()]
        return flower_labels
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{path}' was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")
