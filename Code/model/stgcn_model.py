import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0), stride=(stride,1))
        self.relu = nn.ReLU()

        self.down = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1))

    def forward(self, x):
        y = self.tcn(self.gcn(x)) + self.down(x)
        return self.relu(y)

class STGCN(nn.Module):
    def __init__(self, num_classes=60):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(3 * 25 * 2)

        self.layer1 = STGCNBlock(3, 64, stride=1)
        self.layer2 = STGCNBlock(64, 64)
        self.layer3 = STGCNBlock(64, 64)
        self.layer4 = STGCNBlock(64, 64)
        self.layer5 = STGCNBlock(64, 128, stride=2)
        self.layer6 = STGCNBlock(128, 128)
        self.layer7 = STGCNBlock(128, 128)
        self.layer8 = STGCNBlock(128, 256, stride=2)
        self.layer9 = STGCNBlock(256, 256)
        self.layer10 = STGCNBlock(256, 256)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 1, 4, 3, 2).contiguous()  # (N,C,M,V,T)
        x = x.view(N, C * V * M, T)
        x = self.data_bn(x)
        x = x.view(N, C, M, V, T).permute(0,1,4,3,2).contiguous()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        out = F.avg_pool2d(x, x.size()[2:])
        out = out.view(N, -1)
        out = self.fc(out)
        return out
