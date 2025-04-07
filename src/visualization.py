import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from skimage.transform import resize

def denormalize(img_tensor, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    """
    Reverse the normalization process to convert tensors back to displayable images
    """
    img = img_tensor.numpy().transpose((1, 2, 0)) 
    mean = np.array(normalize_mean)
    std = np.array(normalize_std)
    img = std * img + mean
    return np.clip(img, 0, 1)

def visualize_dataset_samples(dataset, class_names, num_samples=5, save_path=None, show_augmented=True):
    """
    Visualize sample images from the dataset with multiple visualization options
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset to visualize samples from
    class_names : list
        List of class names corresponding to the labels
    num_samples : int, optional
        Number of samples to visualize (default=5)
    save_path : str, optional
        If provided, saves the visualization to this path instead of displaying
    show_augmented : bool, optional
        If True, also shows augmented versions of training data
    """
    # Determine if we're dealing with training data (has augmentations)
    is_training = any(isinstance(t, transforms.RandomResizedCrop) for t in dataset.transform.transforms)
    
    # Create figure with appropriate size and subplots
    rows = 3 if (is_training and show_augmented) else 2
    fig, axes = plt.subplots(rows, num_samples, figsize=(num_samples * 3, rows * 2.5))
    
    # Save original transform
    original_transform = dataset.transform
    
    # For raw images (no normalization)
    raw_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),  # Added to ensure consistent size
        transforms.ToTensor()
    ])
    
    # For seeing augmentation effects (if training data)
    aug_transform = None
    if is_training and show_augmented:
        # Extract just the augmentation transforms (without normalization)
        aug_transforms = [t for t in dataset.transform.transforms 
                         if not isinstance(t, (transforms.Normalize, transforms.ToTensor))]
        aug_transforms.append(transforms.ToTensor())
        aug_transform = transforms.Compose(aug_transforms)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Get sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Store images and labels
    normalized_imgs = []
    raw_imgs = []
    augmented_imgs = []
    labels = []
    
    # Temporarily override the dataset's __getitem__ to record indices
    original_getitem = dataset.__getitem__
    
    try:
        # Collect samples
        for idx in indices:
            # Get normalized image with original transform
            dataset.transform = original_transform
            img, label = original_getitem(idx)
            normalized_imgs.append(img)
            labels.append(label)
            
            # Get raw image
            dataset.transform = raw_transform
            raw_img, _ = original_getitem(idx)
            raw_imgs.append(raw_img)
            
            # Get augmented image if applicable
            if aug_transform:
                dataset.transform = aug_transform
                # Apply augmentation multiple times and pick one (augmentations are random)
                aug_candidates = [original_getitem(idx)[0] for _ in range(3)]
                # Select the one that differs most from original (for better visualization)
                diffs = [(c - raw_img).abs().mean().item() for c in aug_candidates]
                augmented_imgs.append(aug_candidates[np.argmax(diffs)])
    
    finally:
        # Restore original dataset behavior
        dataset.transform = original_transform
        dataset.__getitem__ = original_getitem
    
    # Plot the images
    for i in range(num_samples):
        # Plot normalized image
        denorm_img = denormalize(normalized_imgs[i].cpu())
        axes[0, i].imshow(denorm_img)
        axes[0, i].set_title("Normalized")
        axes[0, i].axis('off')
        
        # Plot raw image
        raw_img = raw_imgs[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(raw_img)
        class_name = class_names[labels[i]] if labels[i] < len(class_names) else "Unknown"
        axes[1, i].set_title(f"Original: {class_name}")
        axes[1, i].axis('off')
        
        # Plot augmented image if applicable
        if aug_transform:
            aug_img = augmented_imgs[i].cpu().permute(1, 2, 0).numpy()
            axes[2, i].imshow(aug_img)
            axes[2, i].set_title("Augmented")
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def visualize_misclassifications_with_closest_images(model, data_loader, class_names, dataset, num_samples=5, device=None):
    """
    Find and visualize misclassified images, along with the closest examples 
    of both the correct class and predicted class, ensuring diversity of classes
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing test images
        class_names: List of class names
        dataset: The full dataset to find examples from
        num_samples: Number of misclassifications to display
        device: Device to run computations on (inferred from model if None)
    """
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    
    # Determine device if not provided
    if device is None:
        if hasattr(model, 'model'):
            device = next(model.model.parameters()).device
        else:
            device = next(model.parameters()).device
        
    # Set model to evaluation mode
    if hasattr(model, 'model'):
        model.model.eval()
    else:
        model.eval()
        
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_features = []
    
    # Track classes we've already collected to ensure diversity
    seen_true_classes = set()
    seen_class_pairs = set()  # Track (true_class, pred_class) pairs
    
    # Find misclassified images and extract their features
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model outputs
            if hasattr(model, 'model'):
                outputs = model.model(inputs)
            else:
                outputs = model(inputs)
                
            _, preds = torch.max(outputs, 1)
            
            # Get feature representations
            features = get_features(model, inputs, device)
            
            # Find misclassifications
            for i in range(len(inputs)):
                if preds[i] != labels[i]:
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    class_pair = (true_label, pred_label)
                    
                    # Check if we want to include this misclassification for diversity
                    # Either the true class is new, or the (true, pred) pair is new
                    if (true_label not in seen_true_classes or 
                        class_pair not in seen_class_pairs):
                        
                        misclassified_images.append(inputs[i].cpu())
                        misclassified_labels.append(true_label)
                        misclassified_preds.append(pred_label)
                        misclassified_features.append(features[i].cpu())
                        
                        # Update our tracking sets
                        seen_true_classes.add(true_label)
                        seen_class_pairs.add(class_pair)
                        
                        # Break early if we have enough diverse samples
                        if len(misclassified_images) >= num_samples:
                            break
            
            if len(misclassified_images) >= num_samples:
                break
    
    # If we didn't find enough misclassifications, show a warning
    if len(misclassified_images) < num_samples:
        print(f"Warning: Only found {len(misclassified_images)} diverse misclassifications")
    
    # If we didn't find any misclassifications, return
    if not misclassified_images:
        print("No misclassifications found in the batches checked")
        return
        
    # Find closest images for each misclassification
    closest_true_class_images = []
    closest_pred_class_images = []
    
    for idx, (features, true_label, pred_label) in enumerate(zip(misclassified_features, 
                                                                misclassified_labels, 
                                                                misclassified_preds)):
        print(f"Finding closest images for misclassification {idx+1}/{len(misclassified_images)}...")
        
        # Find closest image in true class
        closest_true = find_closest_image(model, dataset, features, true_label, device)
        closest_true_class_images.append(closest_true)
        
        # Find closest image in predicted class
        closest_pred = find_closest_image(model, dataset, features, pred_label, device)
        closest_pred_class_images.append(closest_pred)
    
    # Plot misclassified images along with closest true class and predicted class examples
    fig, axes = plt.subplots(3, len(misclassified_images), figsize=(len(misclassified_images) * 3, 9))
    
    # Handle case with only one misclassification
    if len(misclassified_images) == 1:
        axes = np.array(axes).reshape(3, 1)
    
    for i, (wrong_img, true_label, pred_label, true_img, pred_img) in enumerate(
            zip(misclassified_images, misclassified_labels, misclassified_preds, 
                closest_true_class_images, closest_pred_class_images)):
        
        # Plot closest true class image (row 0)
        true_example_img = denormalize(true_img)
        axes[0, i].imshow(true_example_img)
        axes[0, i].set_title(f"Closest in true class:\n{class_names[true_label]}")
        axes[0, i].axis('off')
        
        # Plot misclassified image (row 1)
        wrong_img_display = denormalize(wrong_img)
        axes[1, i].imshow(wrong_img_display)
        axes[1, i].set_title(f"Misclassified\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        axes[1, i].axis('off')
        
        # Plot closest predicted class image (row 2)
        pred_example_img = denormalize(pred_img)
        axes[2, i].imshow(pred_example_img)
        axes[2, i].set_title(f"Closest in pred class:\n{class_names[pred_label]}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Misclassification Analysis with Most Similar Images", y=1.02, fontsize=16)
    plt.show()
    
    # Return the diversity stats for reference
    return {
        "unique_true_classes": len(seen_true_classes),
        "unique_class_pairs": len(seen_class_pairs),
        "total_examples": len(misclassified_images)
    }

# Helper function to extract features from the model
def get_features(model, inputs, device=None):
    """
    Extract features from the penultimate layer of the model
    """
    # Determine device if not provided
    if device is None:
        if hasattr(model, 'model'):
            device = next(model.model.parameters()).device
        else:
            device = next(model.parameters()).device
        
    # Ensure tensors are on correct device
    inputs = inputs.to(device)
    
    # Create feature extractor based on model structure
    if hasattr(model, 'model'):
        feature_extractor = copy.deepcopy(model.model)
    else:
        feature_extractor = copy.deepcopy(model)
        
    # Move to appropriate device
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  # Set to evaluation mode

    # Remove the final fully connected layer
    if hasattr(feature_extractor, 'fc') or hasattr(feature_extractor, 'fc2'):    
        # For ResNet
        feature_extractor.fc = nn.Identity()
    elif hasattr(feature_extractor, 'classifier'):
        # For VGG, DenseNet, etc.
        if isinstance(feature_extractor.classifier, nn.Sequential):
            feature_extractor.classifier = nn.Sequential(
                *list(feature_extractor.classifier.children())[:-1]
            )
        else:
            feature_extractor.classifier = nn.Identity()
    elif hasattr(feature_extractor, 'head'):
        # For Vision Transformer models
        feature_extractor.head = nn.Identity()
    else:
        raise ValueError("Unsupported model architecture for feature extraction")
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(inputs)
    
    return features

# Helper function to find the closest image in a dataset to a given feature vector
def find_closest_image(model, dataset, target_features, target_class, device=None):
    """
    Find the image in the dataset with the closest feature representation to target_features
    and belonging to target_class
    """
    # Determine device if not provided
    if device is None:
        if hasattr(model, 'model'):
            device = next(model.model.parameters()).device
        else:
            device = next(model.parameters()).device
            
    closest_img = None
    min_distance = float('inf')
    
    # Create a smaller dataset with just the target class to speed up search
    target_indices = [i for i in range(len(dataset)) if dataset[i][1] == target_class]
    
    # Check if we found any samples of the target class
    if not target_indices:
        print(f"Warning: No samples found for class {target_class}")
        # Return a blank image with appropriate shape as a fallback
        dummy_img, _ = dataset[0]
        return torch.zeros_like(dummy_img)
    
    # Process in batches to speed up computation
    batch_size = 32
    
    # Move target_features to the same device as the model
    target_features = target_features.to(device)
    # Normalize target feature vector (ensure it's a 1D tensor)
    target_features = target_features.view(1, -1)
    target_features_normalized = F.normalize(target_features, p=2, dim=1)
    
    for i in range(0, len(target_indices), batch_size):
        batch_indices = target_indices[i:i+batch_size]
        
        # Load batch images
        batch_imgs = []
        for idx in batch_indices:
            img, _ = dataset[idx]
            batch_imgs.append(img)
        
        # Convert to tensor and move to device
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # Extract features
        with torch.no_grad():
            batch_features = get_features(model, batch_tensor, device)
        
        # Ensure batch_features is 2D (batch_size × feature_dim)
        batch_features = batch_features.view(batch_features.size(0), -1)
        
        # Compute cosine similarity
        batch_features_normalized = F.normalize(batch_features, p=2, dim=1)
        similarities = torch.mm(
            batch_features_normalized, 
            target_features_normalized.t()
        ).squeeze()
        
        # Find most similar in this batch
        if similarities.dim() == 0:  # Handle case with just one example
            batch_max_sim = similarities
            batch_max_idx = 0
        else:
            batch_max_sim, batch_max_idx = similarities.max(0)
            
        similarity = batch_max_sim.item()
        distance = 1 - similarity  # Convert similarity to distance
        
        if distance < min_distance:
            min_distance = distance
            closest_img = batch_imgs[batch_max_idx].cpu()  # Move back to CPU for storing
    
    return closest_img


def plot_top_10_confusion_matrix(y_true, y_pred, class_names=None, top_n=10, figsize=(12, 8)):
    """
    Plot confusion matrix for the top N most frequent classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_n: Number of top features to plot (default is 10)
        figsize: Size of the plot
    """
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Sum each row to get the total occurrences of each class
    class_totals = np.sum(cm, axis=1)
    
    # Get indices of the top N most frequent classes based on total occurrences
    top_n_indices = np.argsort(class_totals)[-top_n:][::-1]  # Sort and get top n classes
    
    # Filter the confusion matrix for the top N classes
    cm_top_n = cm[top_n_indices, :][:, top_n_indices]
    
    # Get the class names for the top N classes
    top_n_class_names = [class_names[i] for i in top_n_indices]
    
    # Plot the confusion matrix for the top N classes
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm_top_n, annot=True, fmt="d", cmap="Blues", xticklabels=top_n_class_names, yticklabels=top_n_class_names, 
                     cbar_kws={'label': 'Number of Predictions'}, annot_kws={"size": 10})
    
    # Rotate axis labels for readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Set labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Top {top_n} Classes')
    
    # Tight layout to avoid clipping
    plt.tight_layout()
    plt.show()

    # Print classification report for top N classes
    print(classification_report(y_true, y_pred, target_names=top_n_class_names))

def generate_cam(model, image, target_class, device):
    """
    Generate Class Activation Map for a model and image.
    This is a simple implementation and may need adjustment for different models.
    
    Args:
        model: Trained model
        image: Input image tensor [1, 1, H, W]
        target_class: Target class index
        device: Device to run model on
    
    Returns:
        cam: Class activation map
        output: Model output
    """
    # Set model to evaluation mode
    model.eval()
    
    # First, remove any existing hooks to avoid conflicts
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
    
    # Move image to device
    image = image.to(device)
    
    # Forward pass
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach())
    
    # Find and register hook on the last convolutional layer
    hook_handle = None
    
    if False:
        hook_handle = model.conv3.register_forward_hook(hook_fn)
    else:
        # Generic hook on the last conv layer (may need adjustment)
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is not None:
            hook_handle = last_conv.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = model(image)
    
    # Remove the hook after use
    if hook_handle is not None:
        hook_handle.remove()
    
    # Get weights for the target class
    fc_weights = model.fc.weight[target_class].cpu().data.numpy()
    
    # Get feature maps from the last convolutional layer
    feature_maps = features[0].cpu().data.numpy().squeeze()
    
    # Use the first C weights where C is the number of channels
    num_channels = feature_maps.shape[0]
    
    # Option 1: Use a slice of weights if we have more weights than channels
    if len(fc_weights) > num_channels:
        channel_weights = fc_weights[:num_channels]
    # Option 2: If weights are exactly for reshaping (which they should be)
    elif len(fc_weights) == num_channels * feature_maps.shape[1] * feature_maps.shape[2]:
        weights_reshaped = fc_weights.reshape(num_channels, feature_maps.shape[1], feature_maps.shape[2])
        channel_weights = weights_reshaped.mean(axis=(1, 2))
    # Option 3: Fallback - use equal weights
    else:
        channel_weights = np.ones(num_channels)
    
    # Calculate the weighted sum of feature maps
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(channel_weights):
        cam += w * feature_maps[i]
    
    # Apply ReLU to the CAM
    cam = np.maximum(cam, 0)
    
    # Normalize CAM
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)  # Added epsilon to avoid division by zero
    

    cam = resize(cam, (28, 28))
    
    return cam, output


import matplotlib.pyplot as plt
import torch
import random

import random

def show_cam(model, data_loader, device, class_names, num_images=5):
    """
    Show Class Activation Maps for a few images.
    
    Args:
        model: Trained model
        data_loader: DataLoader with data
        device: Device to run model on
        class_names: List of class names
        num_images: Number of images to show
    """
    # Get a list of all the images and labels from the data_loader
    all_images = []
    all_labels = []
    
    for images, labels in data_loader:
        all_images.append(images)
        all_labels.append(labels)
    
    # Stack images and labels into single tensors
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Randomly select `num_images` indices from the dataset
    random_indices = random.sample(range(0, len(all_images)), num_images)
    print(f"Random indices: {random_indices}, len(images): {len(all_images)}")
    
    plt.figure(figsize=(15, num_images * 3))
    
    for i, idx in enumerate(random_indices):
        # Correctly select a single image using idx
        image = all_images[idx]
        label = all_labels[idx].item()
        
        # Generate CAM
        cam, output = generate_cam(model, image.unsqueeze(0), label, device)
        
        # Get predicted class
        _, pred = torch.max(output, 1)
        pred = pred.item()
        
        # Plot original image - handling RGB properly
        plt.subplot(num_images, 3, i*3 + 1)
        img = image.cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)
        
        # Rescale image to [0, 1] if needed, as images are typically in [-1, 1] range after normalization
        img = (img * 0.5) + 0.5  # Rescale to [0, 1]
        
        plt.imshow(img)  # No need for a colormap for RGB images
        plt.title(f"Original Image\nTrue: {class_names[label]}")
        plt.axis('off')
        
        # Plot CAM
        plt.subplot(num_images, 3, i*3 + 2)
        plt.imshow(cam, cmap='jet')
        plt.title("Class Activation Map")
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(num_images, 3, i*3 + 3)
        plt.imshow(img)  # Use the RGB image
        plt.imshow(cam, alpha=0.5, cmap='jet')  # Overlay CAM with alpha blending
        plt.title(f"Overlay\nPred: {class_names[pred]}")
        plt.axis('off')
    
    plt.suptitle(f"Class Activation Maps for {model.__class__.__name__}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()
