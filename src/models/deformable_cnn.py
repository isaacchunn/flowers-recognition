import torch.nn as nn
import torchvision.ops as ops

class DeformableCNN(nn.Module):
    """
    CNN model with deformable convolutions for image classification
    """
    def __init__(self, model_name="deformable_cnn", num_classes=102):
        super(DeformableCNN, self).__init__()
        
        self.model_name = model_name
        
        """
        Image size is rescaled to 224 in our transforms so we can use a 224x224 input size
        First block is used for initial feature extraction to allow network to learn basic features early on (input channels is 3 due to rgb)
        First conv layer has filter size of 32, followed by a batch normalization layer and a second conv layer with filter size of 64 for better feature representation
        Then use max pooling to reduce the spatial dim and preserve the features
        Use standard conv first as deformable are more beneficial in higher layers I think
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
        Use standard conv first as deformable are more beneficial in higher layers I think
        """
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        # Offset predictor for deformable conv in block3, define it for deformconv2d
        self.offset3 = nn.Conv2d(256, 2*3*3, kernel_size=3, padding=1)  # 2*k*k offsets
        self.deform_conv3_2 = ops.DeformConv2d(256, 512, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # For Block 4, we'll likewise use deformable conv after calculating the offset
        # so that is can capture the complex spatial relationship to hopefully handle objects
        # with rigid structure or varying poses
        self.offset4 = nn.Conv2d(512, 2*3*3, kernel_size=3, padding=1)
        self.deform_conv4 = ops.DeformConv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
     
        self.block5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1 and 2
        x = self.block1(x)
        x = self.block2(x)
        
        # Block 3 with deformable conv
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)
        
        # Compute offsets for deformable conv
        offset3 = self.offset3(x)
        # Apply deformable conv
        x = self.deform_conv3_2(x, offset3)
        x = self.bn3_2(x)
        x = self.relu3_2(x)
        x = self.pool3(x)
        
        # Block 4 with deformable conv
        offset4 = self.offset4(x)
        x = self.deform_conv4(x, offset4)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Rest of the network
        x = self.block5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x