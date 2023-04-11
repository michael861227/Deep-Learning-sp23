import torch.nn as nn 

class DeepConvNet(nn.Module):
    def __init__(self, activation, dropout, channels = [25, 50, 100, 200]):
        super().__init__()
        self.channels = channels 
        
        self.conv_0 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = self.channels[0],
                kernel_size = (1, 5),
                bias = False
            ),

            nn.Conv2d(
                in_channels = self.channels[0],
                out_channels = self.channels[0],
                kernel_size = (2, 1),
                bias = False
            ),

            nn.BatchNorm2d(self.channels[0]),
            activation,
            nn.MaxPool2d(kernel_size = (1, 2)),
            nn.Dropout(p = dropout)
        )
        
        for idx, channel in enumerate(self.channels[:-1], start = 1):
            setattr(self, f'conv_{idx}', 
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels = channel,
                            out_channels = self.channels[idx],
                            kernel_size = (1, 5),
                            bias = False
                        ),
                        nn.BatchNorm2d(self.channels[idx], eps = 1e-5, momentum = 0.1),
                        activation,
                        nn.MaxPool2d(kernel_size = (1, 2)),
                        nn.Dropout(p = dropout)
                    )
            )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 8600, out_features = 2)
        )
    
    def forward(self, input):
        out = input
        for idx in range(len(self.channels)):
            out = getattr(self, f'conv_{idx}')(out)
        
        return self.classify(out)
        