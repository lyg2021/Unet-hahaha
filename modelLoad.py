import torch

# 模型
from unet import *
from SETR.transformer_seg import SETRModel
from DeepLabV3Plus.network.modeling import _segm_resnet

def Model_Load(model_name:str):

    if model_name == "unet":
        model = Unet()
        return model

    elif model_name == "setr":
        model = SETRModel(patch_size=(32, 32),
                        in_channels=3,
                        out_channels=1,
                        hidden_size=1024,
                        num_hidden_layers=8,
                        num_attention_heads=16,
                        decode_features=[512, 256, 128, 64])
        return model
        
    elif model_name == "deeplabv3plus":
        model = _segm_resnet(name="deeplabv3plus",
                         backbone_name="resnet50", 
                         num_classes=1, output_stride=8, 
                         pretrained_backbone=False)
        return model
    
    elif model_name == "deeplabv3":
        model = _segm_resnet(name="deeplabv3",
                         backbone_name="resnet50", 
                         num_classes=1, output_stride=8, 
                         pretrained_backbone=False)
        return model

    else:
        print("无 {} 模型".format(model_name))

