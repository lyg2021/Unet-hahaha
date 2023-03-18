import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from data import MyDataset

transform_to_Tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
"""ToTensor最好不要在对图片处理的时候用，它会自动将图片的像素进行归一化处理"""

transform_to_PILImage = transforms.Compose([
    transforms.ToPILImage()
])
"""相对的ToPILImage会自动将tensor的图片乘255还原，所以输出的tensor值并不是图片原本的像素值"""

if __name__ == "__main__":

    path = r"aeroscapes\SegmentationClass\000001_001.png"

    Image.MAX_IMAGE_PIXELS = None
    # image = cv2.imread(path)
    # print(image.shape)

    image = Image.open(path)

    print("图片通道数：", len(image.split()), image.size)

    image_tensor = transform_to_Tensor(image)
    print(image_tensor, image_tensor.shape)


    ##################################

    # image_array = np.asarray(image) 
    # print(image_array, image_array.shape)
    # print(np.unique(image_array))

    # save_image = Image.fromarray(image_array)

    # save_image.show()

    ##########################################




    # tensor = transform_to_Tensor(image)
    # # tensor *= 255
    # print(tensor.unique())
    # print(tensor,tensor.shape)
    # image_mask_tensor = tensor



    # # print(image_mask_tensor)

    # print(image_mask_tensor)

    # image_show = transform_to_PILImage(image_mask_tensor)
    # image_show.show()