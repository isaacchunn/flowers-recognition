import torch
import torch.nn as nn
import torchvision.models as models

class BasePretrainedModel(nn.Module):
    """
    Class for the baseline model using a pretrained architecture
    """
    def __init__(self, num_classes, freeze_layers=False, model_name='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_layers = freeze_layers
        
        # Initialize the model
        self.model = self._initialize_model()
        
         # Apply freezing if requested
        if freeze_layers:
            self._freeze_layers()
    
    def _initialize_model(self):
        """
        Initialize a pretrained model without any modifications
        """
        # For this baseline, we'll use ResNet50 without any modifications
        # We can replace this with other architectures like VGG, DenseNet, etc (if there is time_)
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
            
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
            
        return model
    
    def _freeze_layers(self):
        """
        Freeze all layers except the final fully connected layer
        """
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the final fully connected layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)
  