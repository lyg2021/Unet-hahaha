import torch

# 模型
from unet import *
from SETR.transformer_seg import SETRModel
from DeepLabV3Plus.network.modeling import _segm_resnet
from DeepLabV3Plus.network.modeling import _segm_hrnet


def Model_Load(model_name: str):
    """ model_name = ['unet', 'setr', 'deeplabv3plus_resnet50', 
    'deeplabv3_resnet50', 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32'] """

    if model_name == "unet":        # 精度可以，参数量和计算量有点大了
        model = Unet()
        return model

    elif model_name == "setr":                  # 没效果，无作用，不知道啥原因，训练的少了？但一直不变，没训练了
        model = SETRModel(patch_size=(32, 32),
                          in_channels=3,
                          out_channels=1,
                          hidden_size=1024,
                          num_hidden_layers=8,
                          num_attention_heads=16,
                          decode_features=[512, 256, 128, 64])
        return model

    elif model_name == "deeplabv3plus_resnet50":        # 效果不错，待分析
        model = _segm_resnet(name="deeplabv3plus",
                             backbone_name="resnet50",
                             num_classes=1, output_stride=8,
                             pretrained_backbone=False)
        return model

    elif model_name == "deeplabv3_resnet50":        # 效果不错，待分析
        model = _segm_resnet(name="deeplabv3",
                             backbone_name="resnet50",
                             num_classes=1, output_stride=8,
                             pretrained_backbone=False)
        return model

    elif model_name == "deeplabv3_hrnetv2_32":      # 换成了512*512的输入，之前都是256*256的，跑完，效果不错
        model = _segm_hrnet(name="deeplabv3",
                            backbone_name="hrnetv2_32",
                            num_classes=1,
                            pretrained_backbone=False)
        return model

    elif model_name == "deeplabv3plus_hrnetv2_32":      # 正在跑
        model = _segm_hrnet(name="deeplabv3plus",
                            backbone_name="hrnetv2_32",
                            num_classes=1,
                            pretrained_backbone=False)
        return model

    else:
        print("无 {} 模型".format(model_name))



if __name__ == "__main__":
    """ model_name = ['unet', 'setr', 'deeplabv3plus_resnet50', 
    'deeplabv3_resnet50', 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32'] """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cup")    # 用于测试，有些在gpu上运行发生的错误不显示，放在cpu上就好了-------------------------很管用！
    
    model_name = "deeplabv3plus_hrnetv2_32"

    model = Model_Load(model_name)
    input_tensor = torch.randn(4, 3, 512, 512)

    model, input_tensor = model.to(DEVICE), input_tensor.to(DEVICE)

    output_tensor = model(input_tensor)
    print(model_name, output_tensor.shape)
