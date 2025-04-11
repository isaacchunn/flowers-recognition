import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image
import matplotlib.colors as mcolors
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

def visualize_misclassifications_with_closest_images(model, data_loader, class_names, dataset, num_samples=5, device=None, skip_if_exists=True):
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
    image_path = f"output/diags/misclassification_analysis_{model.model_name}.png"
    # If exist, load the image from directory and show it
    # Check if the image already exists
    if skip_if_exists and os.path.exists(image_path):
        # Load and display the existing image
        img = Image.open(image_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        print(f"Loaded existing image: {image_path}")
        return
    
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
        axes[0, i].set_title(f"Closest in true class:\n{class_names[true_label]} L: {true_label}")
        axes[0, i].axis('off')
        
        # Plot misclassified image (row 1)
        wrong_img_display = denormalize(wrong_img)
        axes[1, i].imshow(wrong_img_display)
        axes[1, i].set_title(f"Misclassified\nTrue: {class_names[true_label]} L: {true_label}\nPred: {class_names[pred_label]} L: {pred_label}")
        axes[1, i].axis('off')
        
        # Plot closest predicted class image (row 2)
        pred_example_img = denormalize(pred_img)
        axes[2, i].imshow(pred_example_img)
        axes[2, i].set_title(f"Closest in pred class:\n{class_names[pred_label]} L: {pred_label}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Misclassification Analysis with Most Similar Images {(model.model_name)}", y=1.02, fontsize=16)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
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
    elif hasattr(model, 'vit'):
        feature_extractor = copy.deepcopy(model.vit)
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

def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history (dict): Dictionary containing training metrics with keys:
                       'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path (str, optional): Path to save the figure. If None, the figure is only displayed.
    
    Returns:
        None: Displays the plot and optionally saves it to the specified path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
def plot_top_n_confusion_matrix(y_true, y_pred, class_names=None, top_n=10, figsize=(12, 8)):
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

    # Filter the true and predicted labels to include only the top N classes
    # Create masks for the top N classes
    top_n_mask = np.isin(y_true, top_n_indices)
    
    # Filter labels
    y_true_filtered = np.array(y_true)[top_n_mask]
    y_pred_filtered = np.array(y_pred)[top_n_mask]
    
    # Map the class indices to a range from 0 to top_n-1 for the classification report
    index_map = {idx: i for i, idx in enumerate(top_n_indices)}
    y_true_mapped = np.array([index_map[idx] for idx in y_true_filtered])
    y_pred_mapped = np.array([index_map.get(idx, -1) for idx in y_pred_filtered])
    
    # Filter out any predictions that map to -1 (i.e., predictions not in top_n_indices)
    valid_indices = y_pred_mapped != -1
    y_true_mapped = y_true_mapped[valid_indices]
    y_pred_mapped = y_pred_mapped[valid_indices]
    
    # Print classification report for top N classes
    print(f"Classification Report - Top {top_n} Classes:")
    print(classification_report(y_true_mapped, y_pred_mapped, target_names=top_n_class_names))


def plot_model_comparison(model_names, accuracies, losses, figsize=(16, 7), 
                          acc_color='skyblue', loss_color='salmon', 
                          save_path=None):
    """
    Create separate bar graphs for model accuracy and loss comparison.
    
    Parameters:
    -----------
    model_names : list
        List of model names to display on x-axis
    accuracies : list
        List of accuracy values corresponding to each model (can be tensors)
    losses : list
        List of loss values corresponding to each model (can be tensors)
    figsize : tuple, optional
        Figure size as (width, height) in inches, default (16, 7)
    acc_color : str, optional
        Color for accuracy bars, default 'skyblue'
    loss_color : str, optional
        Color for loss bars, default 'salmon'
    save_path : str, optional
        If provided, saves the figure to this path, default None
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Convert tensors to numpy arrays if needed
    acc_values = []
    for acc in accuracies:
        if isinstance(acc, torch.Tensor):
            acc_values.append(acc.detach().cpu().numpy())
        else:
            acc_values.append(acc)
    
    loss_values = []
    for loss in losses:
        if isinstance(loss, torch.Tensor):
            loss_values.append(loss.detach().cpu().numpy())
        else:
            loss_values.append(loss)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar width
    bar_width = 0.6
    x = np.arange(len(model_names))
    
    # Plot test accuracy as bars
    acc_bars = ax1.bar(x, acc_values, bar_width, color=acc_color)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)  # Accuracy range from 0 to 1
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=12)
    ax1.set_title('Test Accuracy by Model', fontsize=14, fontweight='bold')
    
    # Annotate accuracy bars
    for bar in acc_bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    
    # Plot test loss as bars
    loss_bars = ax2.bar(x, loss_values, bar_width, color=loss_color)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=12)
    ax2.set_title('Test Loss by Model', fontsize=14, fontweight='bold')
    
    # Annotate loss bars
    for bar in loss_bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def plot_comprehensive_model_comparison(df, figsize=(18, 10), save_path=None):
    """
    Create a comprehensive visualization comparing models on accuracy, loss, and training time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'Model Name', 'Accuracy', 'Loss', 'Training Time (s)'
    figsize : tuple, optional
        Figure size as (width, height) in inches
    save_path : str, optional
        If provided, saves the figure to this path
    """
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort by accuracy for better visualization
    df = df.sort_values('Accuracy', ascending=False)
    
    # Create a color palette for different model types
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Create figure with three subplots
    fig = plt.figure(figsize=figsize, dpi=100)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.5])
    
    # Add a title to the entire figure
    fig.suptitle('Model Comparison on Oxford Flowers Dataset', fontsize=20, fontweight='bold', y=0.98)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, :2])  # Accuracy bars
    ax2 = fig.add_subplot(gs[0, 2:])  # Loss bars
    ax3 = fig.add_subplot(gs[1, :])   # Combined scatter plot
    
    # 1. Bar chart for Accuracy
    bars = ax1.bar(df['Model Name'], df['Accuracy'], color=colors[:len(df)], 
                   edgecolor='black', linewidth=1, alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=16, pad=10)
    ax1.set_ylim(0, max(df['Accuracy']) * 1.15)  # Add some headroom
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add value labels to accuracy bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, 
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", 
                                              fc="white", ec="black", alpha=0.8))
    
    # 2. Bar chart for Loss
    bars = ax2.bar(df['Model Name'], df['Loss'], color=colors[:len(df)], 
                   edgecolor='black', linewidth=1, alpha=0.8)
    ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Model Loss', fontsize=16, pad=10)
    ax2.set_ylim(0, max(df['Loss']) * 1.15)  # Add some headroom
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Add value labels to loss bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, 
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", 
                                              fc="white", ec="black", alpha=0.8))
    
    # 3. Scatter plot: Accuracy vs. Training Time with size representing Loss
    scatter = ax3.scatter(df['Training Time (s)'], df['Accuracy'], 
                s=300 / (df['Loss'] + 0.1),  # Inverse of loss for size (larger = better)
                c=range(len(df)),  # Use index for color to match bar charts
                cmap=plt.cm.tab10,
                alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add a trend line
    z = np.polyfit(df['Training Time (s)'], df['Accuracy'], 1)
    p = np.poly1d(z)
    ax3.plot(df['Training Time (s)'], p(df['Training Time (s)']), 
             "r--", alpha=0.7, label=f"Trend: y={z[0]:.2e}x+{z[1]:.2f}")
    
    # Add model name labels to scatter points
    for i, txt in enumerate(df['Model Name']):
        ax3.annotate(txt, (df['Training Time (s)'].iloc[i], df['Accuracy'].iloc[i]),
                    xytext=(7, 0), textcoords='offset points', fontsize=9)
    
    # Scatter plot formatting
    ax3.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax3.set_title('Accuracy vs. Training Time (bubble size represents inverse of loss)', 
                 fontsize=16, pad=10)
    
    # Add legend for the trend line
    ax3.legend(loc='upper right')
    
    # Add gridlines
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Add a note about the bubble size
    ax3.text(0.02, 0.02, "Note: Larger bubbles represent lower loss values", 
             transform=ax3.transAxes, fontsize=10, style='italic')
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import random

def generate_cam_output(model, image, target_class, device):
    """
    This creates a Class Activation Mapping visualization for our custom traditional cnn 
    and vision transformers implmenetation
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to visualize (either CNN or ViT)
    image : torch.Tensor
        Input tensor with shape [1, channels, height, width]
    target_class : int
        The class index to visualize
    device : torch.device
        The device to perform computation on
        
    Returns:
    --------
    heatmap : numpy.ndarray
        The resulting class activation map
    predictions : torch.Tensor
        The model's output predictions
    """
    model.eval()
    
    # Clear any existing hooks
    for m in model.modules():
        if hasattr(m, '_forward_hooks'):
            m._forward_hooks.clear()
    
    image = image.to(device)
    activation_maps = []
    
    # Define hook function to capture activations
    def capture_activations(module, input_tensor, output_tensor):
        activation_maps.append(output_tensor.detach())
    
    # Determine model architecture type
    is_transformer = (hasattr(model, 'blocks') and hasattr(model, 'norm')) or (hasattr(model, 'vit') and hasattr(model.vit, 'encoder'))
    hook = None
    
    # Register the appropriate hook based on model type
    if is_transformer:
        # For transformers, get activations from final attention block
        if hasattr(model, 'blocks'):
            hook = model.blocks[-1].register_forward_hook(capture_activations)
        elif hasattr(model, 'vit') and hasattr(model.vit, 'encoder'):
            hook = model.vit.encoder.layers[-1].register_forward_hook(capture_activations)
    else:
        # For CNNs, find the final convolutional layer
        final_conv_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                final_conv_layer = module
        
        if final_conv_layer:
            hook = final_conv_layer.register_forward_hook(capture_activations)
    
    # Run inference
    with torch.no_grad():
        predictions = model(image)
    
    # Clean up hook
    if hook:
        hook.remove()
    
    # Get classification weights for target class
    if is_transformer:
        # Handle transformer head
        if hasattr(model, 'head'):
            class_weights = model.head.weight[target_class].cpu().data.numpy()
        else:
            # Alternative: find appropriate classification layer
            found_weights = False
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features == predictions.size(1):
                    class_weights = module.weight[target_class].cpu().data.numpy()
                    found_weights = True
                    break
            
            if not found_weights:
                # Fallback to uniform weights
                class_weights = np.ones(activation_maps[0].shape[-1])
    else:
        # Handle CNN classifier
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            class_weights = model.classifier.weight[target_class].cpu().data.numpy()
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            class_weights = model.fc.weight[target_class].cpu().data.numpy()
        else:
            # Find the last linear layer as fallback
            last_fc = None
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    last_fc = module
            
            if last_fc is not None:
                class_weights = last_fc.weight[target_class].cpu().data.numpy()
            else:
                # Last resort: equal weighting
                class_weights = np.ones(activation_maps[0].shape[1])
    
    # Verify we captured features
    if not activation_maps:
        raise RuntimeError("Failed to capture activations. Check hook registration.")
    
    # Process feature maps
    features = activation_maps[0].cpu().data.numpy()
    
    # Handle different architectures' feature shapes
    if is_transformer:
        # Process transformer features (exclude CLS token, reshape to spatial grid)
        batch, tokens, channels = features.shape
        
        # Remove CLS token
        patch_features = features[:, 1:, :]
        
        # Calculate grid dimensions
        grid_dim = int(np.sqrt(tokens - 1))  # Square root of (tokens - CLS token)
        
        # Reshape to spatial grid
        patch_features = patch_features.reshape(batch, grid_dim, grid_dim, channels)
        
        # Convert to channel-first format for consistency
        features = np.transpose(patch_features, (0, 3, 1, 2))
    
    # Remove batch dimension for single image
    features = np.squeeze(features, axis=0)
    
    # Get number of feature channels
    n_channels = features.shape[0]
    
    # Handle weight dimension mismatch
    if len(class_weights) > n_channels:
        # Truncate weights to match channels
        weights = class_weights[:n_channels]
    elif len(class_weights) == n_channels:
        # Perfect match - use as is
        weights = class_weights
    else:
        # Use equal weights when dimensions don't align
        weights = np.ones(n_channels)
    
    # Generate heatmap by combining weighted feature maps
    heatmap = np.zeros(features.shape[1:], dtype=np.float32)
    for i, weight in enumerate(weights):
        heatmap += weight * features[i]
    
    # Apply ReLU to keep only positive activations
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize to [0,1] range
    epsilon = 1e-8  # Prevent division by zero
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + epsilon)
    
    # Resize to match input image dimensions
    heatmap = resize(heatmap, (image.shape[2], image.shape[3]))
    
    return heatmap, predictions


def visualize_cam(model, data_loader, device, class_names, num_images=5):
    """
    Visualizes Class Activation Maps for selected images from a dataset.
    Also shows comparisons between correctly and incorrectly classified images.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to generate visualizations for
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the dataset
    device : torch.device
        Device to perform computation on
    class_names : list
        List of class names for the dataset
    num_images : int, optional (default=5)
        Number of random images to visualize
    """
    # Collect sample images for visualization
    image_batch = []
    label_batch = []
    
    # Limit data collection to avoid memory issues
    sample_limit = num_images * 20  # Increased to find enough misclassified examples
    
    # Extract samples from data loader
    for batch_x, batch_y in data_loader:
        image_batch.append(batch_x)
        label_batch.append(batch_y)
        
        # Stop after collecting enough samples
        if sum(b.size(0) for b in image_batch) >= sample_limit:
            break
    
    # Combine all collected samples
    all_samples = torch.cat(image_batch, dim=0)
    all_labels = torch.cat(label_batch, dim=0)
    
    # Preprocess to find correctly and incorrectly classified images
    correct_indices = []
    incorrect_indices = []
    
    # Run inference on all samples to identify correct/incorrect predictions
    model.eval()
    with torch.no_grad():
        for idx in range(len(all_samples)):
            img = all_samples[idx].unsqueeze(0).to(device)
            true_label = all_labels[idx].item()
            
            # Get model prediction
            output = model(img)
            _, pred = torch.max(output, 1)
            pred_label = pred.item()
            
            # Sort into correct and incorrect predictions
            if pred_label == true_label:
                correct_indices.append(idx)
            else:
                incorrect_indices.append(idx)
    
    # Check if we have enough data
    if len(correct_indices) == 0:
        print("No correctly classified images found in the sample!")
        return
    
    if len(incorrect_indices) == 0:
        print("No incorrectly classified images found. Model might be too accurate for comparison.")
        # We'll continue without comparisons
    
    # Select samples to visualize
    samples_to_show = min(num_images, len(correct_indices))
    selected_indices = random.sample(correct_indices, samples_to_show)
    
    # Try to match each correct example with an incorrect example of the same true class
    comparison_indices = []
    
    for idx in selected_indices:
        true_class = all_labels[idx].item()
        
        # Find incorrect examples of the same class
        matching_incorrect = [i for i in incorrect_indices if all_labels[i].item() == true_class]
        
        if matching_incorrect:
            # Use a matching incorrect example
            comparison_indices.append(random.choice(matching_incorrect))
        elif incorrect_indices:
            # Fall back to any incorrect example
            comparison_indices.append(random.choice(incorrect_indices))
        else:
            # No incorrect examples available
            comparison_indices.append(None)
    
    # Log selection information
    print(f"Selected {samples_to_show} correctly classified images")
    print(f"Found matching incorrect examples for {sum(1 for x in comparison_indices if x is not None)} cases")
    
    # Create visualization figure (now with 4 columns)
    fig = plt.figure(figsize=(20, samples_to_show * 4))
    
    # Process each selected image
    for i, (idx, comp_idx) in enumerate(zip(selected_indices, comparison_indices)):
        # Get the current image and its true label
        current_img = all_samples[idx]
        true_label = all_labels[idx].item()
        
        # Try to generate CAM with error handling
        try:
            # Generate activation map for the correct example
            heatmap, predictions = generate_cam_output(model, current_img.unsqueeze(0), true_label, device)
            
            # Get the model's prediction
            _, predicted_class = torch.max(predictions, 1)
            predicted_label = predicted_class.item()
            
            # Row 1: Original image (correct prediction)
            ax1 = plt.subplot(samples_to_show, 4, i*4 + 1)
            
            # Display image based on channels
            if current_img.shape[0] == 3:  # Color image
                # Convert from CxHxW to HxWxC format
                display_img = current_img.cpu().numpy().transpose(1, 2, 0)
                
                # Normalize if needed
                if display_img.min() < 0 or display_img.max() > 1:
                    display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
                
                ax1.imshow(display_img)
            else:  # Grayscale image
                display_img = current_img.cpu().numpy().squeeze()
                ax1.imshow(display_img, cmap='gray')
            
            ax1.set_title(f"Original (Correct)\nTrue: {class_names[true_label]}")
            ax1.axis('off')
            
            # Row 2: Activation heatmap for correct prediction
            ax2 = plt.subplot(samples_to_show, 4, i*4 + 2)
            ax2.imshow(heatmap, cmap='inferno')
            ax2.set_title("Activation Heatmap")
            ax2.axis('off')
            
            # Row 3: Overlay visualization for correct prediction
            ax3 = plt.subplot(samples_to_show, 4, i*4 + 3)
            
            # Base image
            if current_img.shape[0] == 3:  # Color
                ax3.imshow(display_img)
            else:  # Grayscale
                ax3.imshow(display_img, cmap='gray')
            
            # Add heatmap overlay
            ax3.imshow(heatmap, alpha=0.6, cmap='inferno')
            ax3.set_title(f"Overlay\nPred: {class_names[predicted_label]} ✓")
            ax3.axis('off')
            
            # Row 4: Comparison with an incorrectly classified image
            ax4 = plt.subplot(samples_to_show, 4, i*4 + 4)
            
            if comp_idx is not None:
                # Get the comparison (incorrect) image
                comp_img = all_samples[comp_idx]
                comp_true_label = all_labels[comp_idx].item()
                
                # Generate CAM for incorrect example
                comp_heatmap, comp_pred = generate_cam_output(model, comp_img.unsqueeze(0), comp_true_label, device)
                _, comp_pred_class = torch.max(comp_pred, 1)
                comp_pred_label = comp_pred_class.item()
                
                # Display comparison image with overlay
                if comp_img.shape[0] == 3:  # Color
                    comp_display = comp_img.cpu().numpy().transpose(1, 2, 0)
                    if comp_display.min() < 0 or comp_display.max() > 1:
                        comp_display = (comp_display - comp_display.min()) / (comp_display.max() - comp_display.min() + 1e-8)
                    ax4.imshow(comp_display)
                else:  # Grayscale
                    comp_display = comp_img.cpu().numpy().squeeze()
                    ax4.imshow(comp_display, cmap='gray')
                
                # Add heatmap overlay
                ax4.imshow(comp_heatmap, alpha=0.6, cmap='inferno')
                
                # Show if it's from the same class or different class
                same_class = comp_true_label == true_label
                class_note = "Same class" if same_class else "Different class"
                ax4.set_title(f"Misclassified ({class_note})\nTrue: {class_names[comp_true_label]}\nPred: {class_names[comp_pred_label]} ✗")
                ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, "No misclassified\nexamples found", 
                        ha='center', va='center')
                ax4.axis('off')
            
        except Exception as error:
            # Handle visualization errors
            err_ax = plt.subplot(samples_to_show, 4, i*4 + 1)
            err_ax.text(0.5, 0.5, f"Visualization failed:\n{str(error)}", 
                      ha='center', va='center', color='red')
            err_ax.axis('off')
    
    # Add model information
    model_name = model.__class__.__name__
    plt.suptitle(f"Class Activation Map Analysis - {model_name}", fontsize=16)
    
    # Layout optimization
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3)
    
    # Show visualization
    plt.show()
