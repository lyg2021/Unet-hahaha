import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

transform_to_Tensor = transforms.Compose([
    transforms.ToTensor()
])

transform_to_PILImage = transforms.Compose([
    transforms.ToPILImage()
])


if __name__ == "__main__":

    path_image = r"aeroscapes\JPEGImages\000001_001.jpg"        # 原图
    path_seg = r"aeroscapes\SegmentationClass\000001_001.png"   # 标签图

    Image.MAX_IMAGE_PIXELS = None       # 去掉最大限制，对大图片用，这里没啥用

    image = Image.open(path_image)
    seg = Image.open(path_seg)


    image_tensor = transform_to_Tensor(image)
    seg_tensor = transform_to_Tensor(seg)

    seg_tensor *= 255                   # 将归一化后的灰度图还原，方便观察

    seg_tensor = seg_tensor.repeat(3, 1, 1)  # 将第一个维度复制三次，后面两维度保持不变

    print(seg_tensor,seg_tensor.shape)

    _image = image_tensor
    _segment_image = seg_tensor
    _output_image = seg_tensor
    

    # 将三张图拼接起来对比看效果
    view_image = torch.stack([_image, _segment_image, _output_image])

    print(view_image, view_image.shape)

    torchvision.utils.save_image(view_image, "stack_image_test.jpg")
    # image_show = transform_to_PILImage(view_image)
    # image_show.show()

