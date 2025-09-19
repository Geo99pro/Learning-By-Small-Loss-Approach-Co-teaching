import os
import torch
import torch.nn as nn
import torchvision.models as models

from torchinfo import summary

os.environ["OMP_NUM_THREADS"] = "1"

def _resolve_weights(model_name: str, model_weight: str):
    if model_name == 'resnet50':
        wt = getattr(models, 'ResNet50_Weights')
    elif model_name == 'vgg16':
        wt = getattr(models, 'VGG16_Weights')
    else:
        raise ValueError(f"Model {model_name} is not supported. Supported models are: vgg16, resnet50.")
    
    return getattr(wt, model_weight)


class GetModel(nn.Module):
    def __init__(self, path_to_save_model_weight: str,
                model_name: str = 'vgg16',
                model_weight: str = 'IMAGENET1K_V1',
                display_model_summary: bool = False
                ) -> nn.Module:
        super().__init__()

        self.path_to_save_model_weight=path_to_save_model_weight
        self.model_name=model_name
        self.model_weight=model_weight
        self.display_model_summary=display_model_summary
        self._process_folder()

        self.backbone, self.feature_dim = self._build_backbone()

        if self.display_model_summary:
            summary(self.backbone, input_size=(1, 3, 224, 224))

    def _process_folder(self):
        if self.path_to_save_model_weight is None:
            folder_path = os.path.join(os.getcwd(), self.model_name)
        else:
            folder_path = os.path.join(self.path_to_save_model_weight, self.model_name)
        os.makedirs(folder_path, exist_ok=True)
        os.environ["TORCH_HOME"] = folder_path

    def _build_backbone(self):
        weights_enum = _resolve_weights(self.model_name, self.model_weight)
        
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=weights_enum)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim
        
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=weights_enum)
            assert isinstance(model.classifier, nn.Sequential)
            feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
            return model, feature_dim
        
        else:
            raise ValueError(f"Model {self.model_name} is not supported. Supported models are: vgg16, resnet50.")

    def get_model(self) -> tuple:
        return self.backbone, self.feature_dim
    
if __name__ == "__main__":
    
    model = GetModel(model_name='vgg16', display_model_summary=True)