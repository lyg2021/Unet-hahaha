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

    # 图片所在路径
    # mask_path = os.path.join("landcover", "ann_dir/test")
    mask_path = r"masks"

    # 获取所有的mask图片名
    name = os.listdir(path=mask_path)

    global unique_cat
    global mask_channels_num_set
    global mask_shape_set

    unique_cat = torch.zeros(1)
    mask_channels_num_set = set()
    mask_shape_set = set()


    # 遍历这些图片，将它们都转化为tensor处理
    for index, image_mask_name in enumerate(name):

        # 通过PIL.Image打开图片
        image_mask = Image.open(os.path.join(mask_path, image_mask_name))

        # 用torchvision.transforms把PIL类型转化为tensor类型
        image_mask_tensor = transform_to_Tensor(image_mask)
        # print(image_mask_tensor, index)
        
        # 单张图片的尺寸获取，加入集合
        image_mask_shape = image_mask.size
        mask_shape_set.add(image_mask_shape)

        # 单张图片的通道数获取，加入集合
        image_mask_channels_num = len(image_mask.split())
        mask_channels_num_set.add(image_mask_channels_num)

        # print(image_mask_tensor.shape)
        # break

        # 获取这张图片中张量中不同的值的张量
        # a = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        # unique = torch.unique(a)
        # print(unique)  # tensor([1, 2, 3, 4])
        unique = torch.unique(image_mask_tensor)

        # 将这个张量和全局变量unique_cat拼接，得到所有图片的不同值的集合
        unique_cat = torch.cat((unique_cat, unique), dim=0)
        print("\r{}/{}:{}".format(index+1, len(name), "▋"*(int((index+1)/len(name)*100)//2)),
              "%{:.1f}".format((index+1)/len(name)*100),
              flush=True,
              end="")
        # sys.stdout.flush()

    # print(unique_cat)
    # 对所有图片的张量的不同值的集合（里面有重复的）再取unique，得到所有图片张量的不同值的张量（没有重复）
    ture_unique = torch.unique(unique_cat)

    # 输出包含所有图片不同张量值的张量（无重复）
    print("\nmask 标签中存在的像素值有：", ture_unique*255)
    print("mask 标签的通道数为：", mask_channels_num_set)
    print("mask 标签的分辨率为：", mask_shape_set)

    # print(image_mask_tensor)

    # image_mask_tensor = image_mask_tensor.masked_fill(image_mask_tensor == 0, 2)

    # print(image_mask_tensor)

    # image_show = transform_to_PILImage(image_mask_tensor)
    # image_show.show()
