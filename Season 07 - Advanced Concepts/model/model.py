import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self, dropout_rate):

        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_rate),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_rate)
        )

        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)  
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
        )

        self.transblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,dilation=2)  
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),

            # Depthwise separable convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=32, padding=1),  
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3)
        )

        self.transblock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  
            #nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)  
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_rate),

        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )  

        self.fc = nn.Sequential(
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        """ This function defines the network structure """

        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)  + x
        x = self.transblock2(x)
        x = self.convblock3(x)  + x
        x = self.transblock3(x) 
        x = self.convblock4(x)  
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x
