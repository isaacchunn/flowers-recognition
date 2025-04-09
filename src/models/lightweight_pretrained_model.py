import torch.nn as nn
import torchvision.models as models
# Import the depthwise seperable conv layer defined in our other source file
from src.layers.depthwise_seperable_layer import DepthwiseSeparableConv

class LightweightPretrainedModel(nn.Module):
    """
    Class for a lightweight model using a pretrained architecture with depthwise+pointwise separable convolutions
    """
    def __init__(self, num_classes, freeze_strategy="layer4", model_name='resnet50'):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
   
        # Initialize the model
        self.model = self._initialize_model()
        
        # Convert to depthwise separable convolutions
        self._convert_to_depthwise_separable()
        
        # Apply initial freezing strategy
        self._apply_freeze_strategy(freeze_strategy)
    
    def _initialize_model(self):
        """
        Initialize a pretrained model without any modifications
        """
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
            
        # Mark the first conv layer to avoid converting it
        model.conv1.is_first_layer = True
            
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
            
        return model
    
    def _convert_to_depthwise_separable(self):
        """
        Convert all standard convolutions to depthwise separable convolutions
        """
        # Skip the first conv layer with special handling for it
        for name, module in self.model.named_children():
            if name != 'conv1':  # Skip the first conv layer
                setattr(self.model, name, convert_to_depthwise_separable(module))
    
    def _apply_freeze_strategy(self, strategy):
        """
        Apply different freezing strategies
        
        Args:
            strategy: String indicating which parts to freeze
                'all': Freeze all layers except the final FC
                'none': Don't freeze any layers (full fine-tuning)
                'partial': Freeze early layers, unfreeze later convolutional layers
                'layer4': Only unfreeze the last convolutional block (layer4) and FC
                'layer3+': Unfreeze layer3, layer4 and FC
        """
        # First, set requires_grad=True for all parameters (default)
        for param in self.model.parameters():
            param.requires_grad = True
            
        if strategy == 'all':
            # Freeze all layers except FC
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
                    
        elif strategy == 'none':
            # Don't freeze any layers (all parameters already have requires_grad=True)
            pass
            
        elif strategy == 'partial':
            # Freeze early convolutional blocks
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
                    param.requires_grad = False
                    
        elif strategy == 'layer4':
            # Freeze all except last conv block and FC
            for name, param in self.model.named_parameters():
                if not any(x in name for x in ['layer4', 'fc']):
                    param.requires_grad = False
                    
        elif strategy == 'layer3+':
            # Freeze only early layers
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
                    param.requires_grad = False
        else:
            raise ValueError(f"Freeze strategy '{strategy}' not recognized")
    
    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)