import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms

transform_to_Tensor = transforms.Compose([
    transforms.ToTensor()
])

transform_to_PILImage = transforms.Compose([
    transforms.ToPILImage()
])


if __name__ == "__main__":

    # 获取所有的mask图片名
    name = os.listdir(os.path.join("aeroscapes", "SegmentationClass_road"))

    global unique_cat

    unique_cat = torch.zeros(1)

    # 遍历这些图片，将它们都转化为tensor处理
    for index, image_mask_name in enumerate(name):

        # 通过PIL.Image打开图片
        image_mask = Image.open(os.path.join("aeroscapes", "SegmentationClass_road", image_mask_name))

        # 用torchvision.transforms把PIL类型转化为tensor类型
        image_mask_tensor = transform_to_Tensor(image_mask)
        # print(image_mask_tensor, index)

        # print(image_mask_tensor.shape)
        # break

        # 获取这张图片中张量中不同的值的张量
        # a = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        # unique = torch.unique(a)
        # print(unique)  # tensor([1, 2, 3, 4])        
        unique = torch.unique(image_mask_tensor)

        # 将这个张量和全局变量unique_cat拼接，得到所有图片的不同值的集合
        unique_cat = torch.cat((unique_cat, unique), dim=0)

    # print(unique_cat)
    # 对所有图片的张量的不同值的集合（里面有重复的）再取unique，得到所有图片张量的不同值的张量（没有重复）
    ture_unique = torch.unique(unique_cat)

    # 输出包含所有图片不同张量值的张量（无重复）
    # tensor([0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431])
    # 共11个类，一个background，总计12个值
    print(ture_unique)

    # print(image_mask_tensor)

    # image_mask_tensor = image_mask_tensor.masked_fill(image_mask_tensor == 0, 2)

    # print(image_mask_tensor)

    # image_show = transform_to_PILImage(image_mask_tensor)
    # image_show.show()