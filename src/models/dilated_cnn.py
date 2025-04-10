import torch.nn as nn

class DilatedCNN(nn.Module):
    """
    This is a balanced CNN model designed for image classification that balances training time and complexity
    with enhanced receptive field using dilated convolutions
    """
    def __init__(self, model_name = "dilated_cnn", num_classes=102):
        super(DilatedCNN, self).__init__()
        
        self.model_name = model_name
        # Input: 3 channels rgb and 224x224 spatial dimensions
        
        """
        Image size is rescaled to 224 in our transforms so we can use a 224x224 input size
        First block is used for initial feature extraction to allow network to learn basic features early on (input channels is 3 due to rgb)
        First conv layer has filter size of 32, followed by a batch normalization layer and a second conv layer with filter size of 64 for better feature representation
        Then use max pooling to reduce the spatial dim and preserve the features
        
        We keep early layers standard to capture the details first before moving the receptive field
        """
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        """
        This second block is used to expand the channel size to 128 and reduce the spatial dimension to 56x56
        This allows the network to learn more complex features while maintaining a reasonable spatial resolution
        """
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        """
        This block is used for more detailed feature extraction, with two convolutional layers to learn more complex features
        We slowly increase the number of channels to get deeper feature extraction
        Add dilation=2 to expand the receptive field
        from 3x3 to 5x5 effective size without additional parameters.
        Increase dilation rate to 4 to expand the receptive field again
        """
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        """
        This block is used to further refine the features extracted in the previous block, with a single convolutional layer 
        and max pooling to reduce the spatial dimension to 14x14
        
        Using huge dilation rate of 6 to capture very wide spatial relationships and global context,
        to model complex patterns
        """
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4),  # Dilation=4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5: Final spatial reduction (14x14 -> 7x7)
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global pooling to reduce the spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers with intermediate FC
        # Classification layer to map the features to the number of classes
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply each block sequentially
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Global pooling and reshape
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification using our intermediate FC
        x = self.classifier(x)
        
        return x
