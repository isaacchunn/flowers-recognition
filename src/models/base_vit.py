import torch
import torch.nn as nn
import torchvision.models as models
import timm

class BaseViT(nn.Module):
    """
    Class for the baseline model using a pretrained architecture
    """
    def __init__(self, num_classes, freeze_layers=True):
        super(BaseViT,self).__init__()
        self.num_classes = num_classes

        # Initialize the model
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True) 

        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

         # Apply freezing if requested
        if freeze_layers:
            self._freeze_layers()
    
    
    def _freeze_layers(self):
        """
        Freeze all layers except the final fully connected layer
        """
        # Freeze vit parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the head for fine-tuning
        for param in self.model.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)
    

