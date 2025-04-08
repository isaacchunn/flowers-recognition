import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """
    This is a balanced CNN model designed for image classification that balances training time and complexity
    """
    def __init__(self, num_classes=102):
        super(BaseCNN, self).__init__()
        
        # Input: 3 channels rgb and 224x224 spatial dimensions
        
        """
        Image size is rescaled to 224 in our transforms so we can use a 224x224 input size
        First block is used for initial feature extraction to allow network to learn basic features early on (input channels is 3 due to rgb)
        First conv layer has filter size of 32, followed by a batch normalization layer and a second conv layer with filter size of 64 for better feature representation
        Then use max pooling to reduce the spatial dim and preserve the features
        """
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        
        """
        This second block is used to expand the channel size to 128 and reduce the spatial dimension to 56x56
        This allows the network to learn more complex features while maintaining a reasonable spatial resolution
        """
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        
        """
        This block is used for more detailed feature extraction, with two convolutional layers to learn more complex features
        We slowly increase the number of channels to get deeper feature extraction
        """
        self.conv3_1 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(192)
        self.conv3_2 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        
        """
        This block is used to further refine the features extracted in the previous block, with a single convolutional layer and max pooling to reduce the spatial dimension to 14x14
        """
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Final spatial reduction to focus on feature presence rather than exact location
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # Global pooling to reduce parameters more compared to flattening
        # Seems to be more parameter efficient than flattening 256x7x7 features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers with intermediate FC
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5) # Just prevent overfitting due to lack of training images per class
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(x)
        
        # Final reduction
        x = self.pool5(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Two-layer classifier
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x