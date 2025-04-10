import torch.nn as nn
import torchvision.models as models
# Import the depthwise seperable conv layer defined in our other source file
from src.layers.depthwise_seperable_layer import DepthwiseSeparableConv

def convert_conv2d_to_depthwise_separable(module, conv_count=None):
    """
    Converts a specific number of the last Conv2d layers to DepthwiseSeparableConv
    
    Args:
        module: The PyTorch module to convert
        conv_count: Number of last Conv2d layers to convert (None means all)
        
    Returns:
        Tuple: (modified module, remaining count to convert, list of converted layer paths)
    """
    # If conv_count is 0, we're done converting
    if conv_count == 0:
        return module, 0, []
    
    # Find all Conv2d layers in the module, including nested layers
    conv_layers = []
    def _find_conv_layers(m, prefix=''):
        for name, child in m.named_children():
            current_path = f"{prefix}.{name}" if prefix else name
            # If this is a Conv2d and not a 1x1 conv and not the first layer
            if isinstance(child, nn.Conv2d):
                is_1x1 = child.kernel_size[0] == 1 and child.kernel_size[1] == 1
                is_first = hasattr(child, 'is_first_layer') and child.is_first_layer
                if not is_1x1 and not is_first:
                    conv_layers.append((current_path, child))
            # Recursively check children
            _find_conv_layers(child, current_path)
    
    _find_conv_layers(module)
    
    # Track converted layer paths
    converted_paths = []
    
    # If we've found Conv2d layers, convert the last N of them based on conv_count
    if conv_layers:
        # Sort to ensure consistent ordering
        conv_layers.sort(key=lambda x: x[0])
        
        # Determine how many layers to convert
        num_to_convert = len(conv_layers) if conv_count is None else min(conv_count, len(conv_layers))
        
        # Convert the last N layers
        for i in range(len(conv_layers) - num_to_convert, len(conv_layers)):
            path, conv = conv_layers[i]
            
            # Create a depthwise separable convolution with the same parameters
            depthwise_separable = DepthwiseSeparableConv(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size[0],  # Assuming square kernels
                stride=conv.stride[0],            # Assuming same stride for both dimensions
                padding=conv.padding[0],          # Assuming same padding for both dimensions
                dilation=conv.dilation[0],        # Assuming same dilation for both dimensions
                bias=conv.bias is not None
            )
            
            # Mark this as a converted layer for freezing purposes
            depthwise_separable.is_converted_layer = True
            
            # Navigate to the parent module and replace the Conv2d
            parent_path, name = path.rsplit('.', 1) if '.' in path else ('', path)
            parent = module
            for part in parent_path.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # Replace the Conv2d with our DepthwiseSeparableConv
            setattr(parent, name, depthwise_separable)
            print(f"Converted layer at {path} to depthwise separable")
            
            # Keep track of converted paths
            converted_paths.append(path)
        
        # Update remaining count
        remaining = 0 if conv_count is None else max(0, conv_count - num_to_convert)
        return module, remaining, converted_paths
    
    # If no Conv2d layers found in this module, return unchanged
    return module, conv_count, []

class LightweightPretrainedModel(nn.Module):
    """
    Class for a lightweight model using a pretrained architecture with depthwise+pointwise separable convolutions
    """
    def __init__(self, num_classes, freeze_strategy="layer4", model_name='resnet50', num_last_conv_layers=None):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.num_last_conv_layers = num_last_conv_layers
        self.converted_layers = []
   
        # Initialize the model
        self.model = self._initialize_model()
        
        # Convert the last N convolutional layers to depthwise separable convolutions
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
        Convert the last N convolutional layers to depthwise separable convolutions
        """
        # Special case - if num_last_conv_layers is 0 or None, don't convert anything
        if self.num_last_conv_layers == 0 or self.num_last_conv_layers is None:
            print("No layers will be converted (num_last_conv_layers is 0 or None)")
            return
            
        print(f"Converting the last {self.num_last_conv_layers} convolutional layers to depthwise separable...")
        
        # Convert the specified number of last Conv2d layers and store the converted layer paths
        self.model, remaining, self.converted_layers = convert_conv2d_to_depthwise_separable(
            self.model, self.num_last_conv_layers
        )
        
        if remaining == 0:
            print("Successfully converted all specified convolutional layers to depthwise separable")
            print(f"Converted {len(self.converted_layers)} layers: {self.converted_layers}")
        else:
            print(f"Warning: Only able to convert {self.num_last_conv_layers - remaining} layers " 
                  f"(requested {self.num_last_conv_layers})")
            print(f"Converted layers: {self.converted_layers}")
    
    def _is_in_converted_path(self, param_name):
        """
        Check if a parameter belongs to a converted layer
        """
        for path in self.converted_layers:
            if path in param_name:
                return True
                
        # Also check for parameters that belong to a module marked as converted
        if '.' in param_name:
            module_path, param = param_name.rsplit('.', 1)
            try:
                # Navigate to the module
                module = self.model
                for part in module_path.split('.'):
                    if part:
                        module = getattr(module, part)
                
                # Check if it's marked as converted
                if hasattr(module, 'is_converted_layer') and module.is_converted_layer:
                    return True
            except (AttributeError, ValueError):
                pass
                
        return False
    
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
                'converted_only': Only train converted layers and FC
                
                Add '_keep_converted' suffix to any strategy to keep converted layers trainable
                Examples: 'all_keep_converted', 'layer4_keep_converted', etc.
        """
        # First, set requires_grad=True for all parameters (default)
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Check if we're using a strategy with the 'keep_converted' modifier
        keep_converted = False
        base_strategy = strategy
        
        if isinstance(strategy, str) and '_keep_converted' in strategy:
            keep_converted = True
            base_strategy = strategy.replace('_keep_converted', '')
            print(f"Using {base_strategy} strategy with converted layers kept trainable")
            
        # Apply the base strategy first
        if base_strategy == 'all':
            # Freeze all layers except FC
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    if not (keep_converted and self._is_in_converted_path(name)):
                        param.requires_grad = False
                    
        elif base_strategy == 'none':
            # Don't freeze any layers (all parameters already have requires_grad=True)
            pass
            
        elif base_strategy == 'partial':
            # Freeze early convolutional blocks
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
                    if not (keep_converted and self._is_in_converted_path(name)):
                        param.requires_grad = False
                    
        elif base_strategy == 'layer4':
            # Freeze all except last conv block and FC
            for name, param in self.model.named_parameters():
                if not any(x in name for x in ['layer4', 'fc']):
                    if not (keep_converted and self._is_in_converted_path(name)):
                        param.requires_grad = False
                    
        elif base_strategy == 'layer3+':
            # Freeze only early layers
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['layer1', 'layer2', 'conv1', 'bn1']):
                    if not (keep_converted and self._is_in_converted_path(name)):
                        param.requires_grad = False
                        
        elif base_strategy == 'converted_only':
            # Special strategy: Only train converted layers and FC
            for name, param in self.model.named_parameters():
                if not ('fc' in name or self._is_in_converted_path(name)):
                    param.requires_grad = False
                    
        else:
            raise ValueError(f"Freeze strategy '{strategy}' not recognized")
        
        # Log trainable parameters (optional - for debugging)
        if hasattr(self, 'verbose') and self.verbose:
            trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
            print(f"Trainable parameters after applying {strategy} strategy:")
            for param_name in trainable_params:
                if self._is_in_converted_path(param_name):
                    print(f"  {param_name} (converted layer)")
                else:
                    print(f"  {param_name}")
    
    def forward(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)
