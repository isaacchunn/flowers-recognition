import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    A CNN model with stacked convolutional layers at each filter interval and batch normalization.
    """
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()

        # First convolutional layer 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
    
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        
        # Second convolutional layer 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56

        # Third convolutional layer 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28

        # Fourth convolutional layer 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  
        self.bn5 = nn.BatchNorm2d(512)  
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # Calculate output size after convolutions and pooling
        # The output after four pooling layers will have the size (512, 14, 14) for an input of size (224, 224)
        self.fc_input_size = 512 * 14 * 14  # Output size after convolutions and pooling
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout rate
        self.fc2 =nn.Linear(512, num_classes)  # Second fully connected layer 
        

    def forward(self, x):
        # First layer (stacked convolutions)
        x = F.relu(self.bn1(self.conv1(x)))                                                                         
        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.pool1(x)
        
        # Second layer
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.pool2(x)
        
        # Third layer
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        
        # Fourth layer
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        
        # Flatten the feature map before passing it to the fully connected layer
        x = x.view(x.size(0), -1) 
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
