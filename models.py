import timm
import torch
from torch import nn

def model(model_name, num_classes=10, pretrained= False, img_size = 64):
    if model_name == 'resnet18':
        model = timm.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'vit_tiny':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes, img_size=img_size)
    elif model_name == 'vit_small':
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes, img_size=img_size)
    
    # initialize all weights for training from scratch
    if not pretrained:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    return model