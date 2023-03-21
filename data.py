import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from PIL import Image


"""VOC 格式数据"""

class MyDataset(Dataset):
    def __init__(self, root_path: str,           # 数据集根目录
                 mask_image_path: str,           # 分割标签目录
                 original_image_path: str,            # 原图目录
                 txt_imageset_path: str,              # 划分训练集、验证集、测试集txt的目录
                 mode: str = "train",                 # 是提取(train)训练集？(val)验证集？还是(test)测试集？默认是train
                 image_size: tuple = (512, 512)    # resize后图片尺寸
                 ) -> None:

        super().__init__()
        self.root_path = root_path
        self.mask_image_path = mask_image_path
        self.original_image_path = original_image_path
        self.txt_imageset_path = txt_imageset_path
        self.image_size = image_size
        self.mode = mode

        # 可以在 transforms.Compose 下对图像做预处理
        self.transforms_original_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),      # 均值mean，（image net中计算的）
                                 (0.2023, 0.1994, 0.2010)),     # 方差std，（image net中计算的）
        ])

        self.transforms_segment_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        if self.mode == "train":
            self.txtfileName = "train.txt"
        elif self.mode == "val":
            self.txtfileName = "val.txt"
        elif self.mode == "test":
            self.txtfileName = "test.txt"
        else:
            print(r"error, 没这个模式, 得输入train/val/test")

        # 获取对应train.txt或val.txt或test.txt里面的名称列表，顺便加个后缀名比如.png
        txt_file = open(os.path.join(self.root_path, self.txt_imageset_path, self.txtfileName), 
                        mode="r",
                        encoding="utf-8")   # 打开这个txt文件，开始读内容
        
        # 读所有行，所有行共同构成一个列表list
        txt_lines = txt_file.readlines()

        # 把读到的换行符搞掉，加上后缀名
        for index, line in enumerate(txt_lines):
            txt_lines[index] = line.replace("\n", "") + ".png"
        
        self.txt_name_lines = txt_lines


    def __len__(self):
        return len(self.txt_name_lines)

    def __getitem__(self, index):

        # 单张分割标签图片的名称，形式为 xxx.png
        segment_name = self.txt_name_lines[index]

        # 单张分割标签的路径
        segment_path = os.path.join(
            self.root_path, self.mask_image_path, segment_name)

        # 单张原图的路径，图片名称后缀是jpg
        image_path = os.path.join(
            self.root_path, self.original_image_path, segment_name.replace("png", "jpg"))

        # 将原图和标签图都处理为
        # image = keep_image_size_open_Image(image_path, self.image_size)
        # segment_image = keep_image_size_open_Segment(segment_path, self.image_size)

        image = Image.open(image_path)
        segment_image = Image.open(segment_path)

        # 将图像转化为 tensor
        image_tensor = self.transforms_original_image(image)
        # print(image_tensor.shape)
        segment_image_tensor = self.transforms_segment_image(segment_image)
        # print(segment_image_tensor.shape)

        return image_tensor, segment_image_tensor


if __name__ == "__main__":
    md = MyDataset(root_path=r"aeroscapes",
                    mask_image_path=r"SegmentationClass_road",
                    original_image_path=r"JPEGImages",
                    txt_imageset_path=r"ImageSets",
                    mode="val")
    print(len(md))

    print(md[0][1])

    # print(md.txt_name_lines)

    print(torch.unique(md[0][1]))

    # print(type(md.txt_name_lines))

    # print(md.txt_name_lines)

    # # len(md)，len调用了self.__len__()
    # print(len(md))

    # # md[500][0]，通过索引可以直接调用self.__getitem__()函数，
    # # print(type(md[500]))  # <class 'tuple'> 直接md[500]返回一个元组
    # # 第二个维度是返回元组的索引
    # print(md[500][0].shape)
    pass
