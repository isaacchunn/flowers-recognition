import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from tqdm import tqdm
import torchvision.models as models

class BasePretrainedModel(nn.Module):
    """
    Class for the baseline model using a pretrained architecture
    """
    def __init__(self, num_classes, device, model_name='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize the model
        self.model = self._initialize_model()
        self.model = self.model.to(device)  # Move to GPU if available

    
    def _initialize_model(self):
        """
        Initialize a pretrained model without any modifications
        """
        # For this baseline, we'll use ResNet50 without any modifications
        # We can replace this with other architectures like VGG, DenseNet, etc
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Replace the final fully connected layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # Replace the final classifier
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            # Replace the final classifier
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_classes)
            
        return model
    
    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)
   
    def save_model(self, path):
        """
        Save the model to disk
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """
        Load a saved model from disk
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
  