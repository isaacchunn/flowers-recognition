import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """
    This is a balanced CNN model designed for image classification that balances training time and complexity
    """
    def __init__(self, model_name = "base_cnn", num_classes=102):
        super(BaseCNN, self).__init__()
        
        self.model_name = model_name
        # Input: 3 channels rgb and 224x224 spatial dimensions
        
        """
        Image size is rescaled to 224 in our transforms so we can use a 224x224 input size
        First block is used for initial feature extraction to allow network to learn basic features early on (input channels is 3 due to rgb)
        First conv layer has filter size of 32, followed by a batch normalization layer and a second conv layer with filter size of 64 for better feature representation
        Then use max pooling to reduce the spatial dim and preserve the features
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
        """
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        """
        This block is used to further refine the features extracted in the previous block, with a single convolutional layer and max pooling to reduce the spatial dimension to 14x14
        """
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5: Final spatial reduction (14x14 -> 7x7) <- do we really need this?
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global pooling to reduce the spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers with intermediate FC
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

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



class BASENN_tripletLoss(nn.Module):
    """
    This is a balanced CNN model designed for image classification that balances training time and complexity
    """
    def __init__(self, model_name = "base_cnn", num_classes=102, embedding_dim=128):
        super(BaseCNN, self).__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        # Input: 3 channels rgb and 224x224 spatial dimensions
        
        """
        Image size is rescaled to 224 in our transforms so we can use a 224x224 input size
        First block is used for initial feature extraction to allow network to learn basic features early on (input channels is 3 due to rgb)
        First conv layer has filter size of 32, followed by a batch normalization layer and a second conv layer with filter size of 64 for better feature representation
        Then use max pooling to reduce the spatial dim and preserve the features
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
        """
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        """
        This block is used to further refine the features extracted in the previous block, with a single convolutional layer and max pooling to reduce the spatial dimension to 14x14
        """
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5: Final spatial reduction (14x14 -> 7x7) <- do we really need this?
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global pooling to reduce the spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # New layer to output embeddings
        self.embedding = nn.Linear(256, self.embedding_dim)
        
        # Classification layers with intermediate FC
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

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
        
        # Output the embedding layer
        embedding = self.embedding(x)
        
        # Classification using our intermediate FC
        x = self.classifier(x)
        
        return x, embedding # Return both the classification output and the embedding
