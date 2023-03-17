from network.modeling import _segm_resnet
import torch

if __name__ == "__main__":
    model = _segm_resnet(name="deeplabv3plus",
                         backbone_name="resnet50", 
                         num_classes=1, output_stride=8, 
                         pretrained_backbone=False)
    
    input_tensor = torch.randn(4, 3, 256, 256)
    output_tensor = model(input_tensor)

    print(output_tensor.shape)