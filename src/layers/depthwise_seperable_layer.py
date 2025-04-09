import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    A module that performs depthwise separable convolution:
    - Depthwise convolution (one filter per input channel)
    - Pointwise convolution (1x1 conv to mix channels)
    
    This should reduce parameter at small loss of accuracy compared to standard conv
    dilation is used to increase the receptive field without increasing the number of parameters
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                dilation=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
