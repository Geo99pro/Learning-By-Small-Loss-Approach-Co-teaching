import os
import torch
import torch.nn as nn
import torchvision.models as models

from nets import GetModel
from torchinfo import summary

class CustomHead(nn.Module):
    def __init__(self, in_features: int,
                 class_number: int,
                 p_drop: float = 0.5) -> nn.Module:
        super().__init__()
        
        self.fc = nn.Linear(in_features, class_number)

    def forward(self, x):
        logits = self.fc(x)
        proba = torch.sigmoid(logits)
        return proba, logits

class CoTeachingNet(GetModel):
    def __init__(self, path_to_model_weight: str = None,
                 model_name: str = 'vgg16',
                 model_weight: str = 'IMAGENET1K_V1',
                 class_number: int = 14,
                 how_many_layers_to_unfreeze: int = 4,
                 display_model_summary: bool = False) -> nn.Module:
        super().__init__(path_to_model_weight, model_name, model_weight, display_model_summary)

        self.how_many_layers_to_unfreeze = how_many_layers_to_unfreeze
        self.backbone, feature_dim = self.get_model()
        self.head = CustomHead(in_features=feature_dim, class_number=class_number)
        self.freeze_model_some_part()

    def forward(self, x):
        features = self.backbone(x)
        probs, logits = self.head(features)
        return probs, logits

    def freeze_model_some_part(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in list(self.parameters())[-self.how_many_layers_to_unfreeze:]:
            param.requires_grad = True

# if __name__ == "__main__":
#     model = CoTeachingNet(model_name='resnet50', display_model_summary=True)
#     image = torch.randn(1, 3, 224, 224)  # Example input tensor
#     summary(model, input_size=(1, 3, 224, 224), device=model.device.type)