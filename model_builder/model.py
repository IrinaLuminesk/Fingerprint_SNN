import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
class Model(nn.Module):
    def __init__(self, embedding_dim = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = self.build_model() 
    def build_model(self):
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, self.embedding_dim)
        return backbone
    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1) # L2 normalization (important)
        return x