import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import shutil
import random


def copy_to_new(old_path: str, new_path: str):
    """脱裤子放屁"""
    shutil.copyfile(old_path, new_path)

transform_to_Tensor = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])
"""ToTensor最好不要在对图片处理的时候用，它会自动将图片的像素进行归一化处理"""

transform_to_PILImage = transforms.Compose([
    transforms.ToPILImage()
])
"""相对的ToPILImage会自动将tensor的图片乘255还原，所以输出的tensor值并不是图片原本的像素值"""

if __name__ == "__main__":

    input_dir = "output"
    output_dir = "output_new"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = os.listdir(input_dir)

    for image_name in image_list:
        if ".png" in image_name:
            # path = os.path.join(input_dir, image_name)
            # # print(os.path.join(root_dir, image_name))

            # image = Image.open(path)
            # # print("图片通道数：", len(image.split()), image.size)

            # image_tensor = transform_to_Tensor(image)
            # # image_tensor *= 255

            # new_image_tensor = image_tensor[0]
            # print(image_tensor.unique())
            # # print(new_image_tensor.unique())
            # # print(new_image_tensor,new_image_tensor.shape)

            # new_image = transform_to_PILImage(new_image_tensor)

            # new_image.save(path.replace(input_dir, output_dir))
            pass
        else:
            path = os.path.join(input_dir, image_name)
            copy_to_new(old_path=path, new_path=path.replace(input_dir, output_dir))

    