import glob
from PIL import Image
import PIL
import cv2
import matplotlib.image as mi
import matplotlib.pyplot as plt
import numpy as np
import png # https://pypng.readthedocs.io/en/latest/
import os


# 使用PIL库生成调色板格式的图
def Convert_Palette(img_png):

    # VOC数据集中的SegmentationClass图像
    voc_label_path = r'D:\研究生\dataset\VOC2007\VOCdevkit\VOC2007\SegmentationClass\000032.png'

    # """test"""
    # voc_image = Image.open(voc_label_path)
    # print(voc_image.size, voc_image.mode)

    # tensor = transform_to_Tensor(voc_image)
    # print(tensor.tolist())

    # 这里可以获取该png图片的信息(tuple格式储存的)，如果是调色板图片，还可以获取调色板
    png_data = png.Reader(voc_label_path)
    # print(png_data.read()) 

    # 得到voc的label的调色板(list形式储存的)
    voc_palette = png_data.read()[3]['palette']     
    # print(voc_palette)
    
    # 将调色板转化为array
    palette = np.array(voc_palette) # 256*3
    # print(palette)

    # 256*1*3 把int32改成uint8(opencv中储存图像的数据格式 0 - 256)
    palette = palette.reshape(256, 1, 3).astype(np.uint8) 
    # print(palette, type(palette))

    # 目标图像
    out_img = Image.open(img_png)

    # 将图像模式变为调色板模式 "P"
    out_img = out_img.convert("P")
    # print(out_img.size, out_img.mode)

    # 将调色板信息添加进目标图像中
    # 就是以刚刚得到的调色板将图片转换为调色板模式的伪彩图
    out_img.putpalette(palette)     

    out_img.save(img_png.replace("SegmentationClass", "SegmentationClass_colorful"))


if __name__ == "__main__":

    # 获取所有的mask图片名
    name = os.listdir(os.path.join("aeroscapes", "SegmentationClass"))

    # 挨个转化为调色板模式的伪彩图
    for index, image_mask_name in enumerate(name):
        image_path = os.path.join("aeroscapes", "SegmentationClass", image_mask_name)
        print(index, image_path)
        Convert_Palette(image_path)


    # Convert_Palette(r"aeroscapes\SegmentationClass\000001_001.png")



