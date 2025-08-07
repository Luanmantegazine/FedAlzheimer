import torch.nn as nn, torchvision.models as models

class ResNet50FL(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x): return self.base(x)