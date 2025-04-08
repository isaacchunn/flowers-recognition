import os
import torch
import json

def save_model_and_history(model, history, save_dir, model_name):
    """
    Save the model state and training history
    
    Args:
        model: Trained PyTorch model
        history: Training history dictionary
        model_name: Name prefix for the saved files
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history as JSON
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    
    # Convert any non-serializable objects (like tensors) to lists or floats
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_history[key] = value
        elif torch.is_tensor(value):
            serializable_history[key] = value.cpu().tolist()
        else:
            serializable_history[key] = [float(v) for v in value]
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    
    print(f"Training history saved to {history_path}")
    
    return model_path, history_path

def load_model_and_history(model_class, save_dir, model_name, device, num_classes=102):
    """
    Load a saved model and its training history
    
    Args:
        model_class: The model class (e.g., BaseCNN)
        model_name: Name prefix of the saved files
        num_classes: Number of classes for model initialization
        
    Returns:
        model: Loaded model
        history: Training history dictionary
    """
    # Define paths
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(history_path):
        print(f"Warning: History file not found at {history_path}")
        history = {}
    else:
        # Load history
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    # Initialize model
    model = model_class(num_classes=num_classes)
    
    # Load state dict with appropriate device mapping
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to the specified device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {model_path}")
    
    return model, history

def check_model_exists(model_name, save_dir):
    """
    Check if a model file exists
    
    Args:
        model_name: Name prefix of the model file
        save_dir: Directory where model is saved
        
    Returns:
        bool: True if model exists, False otherwise
    """
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    return os.path.exists(model_path)
