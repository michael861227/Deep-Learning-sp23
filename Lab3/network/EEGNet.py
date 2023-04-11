import torch.nn as nn 

class EEGNet(nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = (1, 51),
                stride = (1, 1),
                padding = (0, 25),
                bias = False
            ),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats = True)
        )
        
        self.deep_wise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 32, 
                kernel_size = (2, 1), 
                stride = (1, 1), 
                groups = 16, 
                bias = False
            ),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            activation,
            nn.AvgPool2d(kernel_size = (1, 4), stride = (1, 4), padding = 0),
            nn.Dropout(p = dropout)
        )
        
        self.seperable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding = (0, 7),
                bias = False
            ),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            activation,
            nn.AvgPool2d(kernel_size = (1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p = dropout)
        )
        
        self.classify =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 736, out_features = 2, bias = True)
        )
        
    def forward(self, input):
        first_conv_out = self.first_conv(input)
        deep_wise_conv_out = self.deep_wise_conv(first_conv_out)
        separable_conv_out = self.seperable_conv(deep_wise_conv_out)
        out = self.classify(separable_conv_out)
        
        return out