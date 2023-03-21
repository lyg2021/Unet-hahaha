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


def to_road(image_mask_path):

    image_mask = Image.open(image_mask_path)

    image_mask_tensor = transform_to_Tensor(img=image_mask)    

    # 将值不在这个区间的值赋值为某个值
    image_mask_tensor = torch.where(torch.gt(image_mask_tensor, 0.0392) & torch.lt(image_mask_tensor, 0.0393), 1.0, 0)

    # image_mask_tensor = image_mask_tensor / 255
    
    # image_mask_tensor *= 255
    # image_mask_tensor = torch.where(torch.gt())

    # 如果 tensor 中的值全等于0 则输出0
    if not (image_mask_tensor!=0).any():
        print("0")

    image_show = transform_to_PILImage(image_mask_tensor)

    # 全等于 0 则不保存
    if (image_mask_tensor!=0).any():
        image_show.save(image_mask_path.replace("SegmentationClass", "SegmentationClass_road"))

    # image_show.show()


    # tensor([0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275, 0.0314,
    #     0.0353, 0.0392, 0.0431])


    """
    0.0392 道路
    0.0039 人
    0.0431 天空
    """

"""
a = torch.rand((2, 3))
# tensor([[0.2620, 0.4850, 0.5924],
#         [0.4152, 0.0475, 0.5491]])

# tensor中在区间[0.3, 0.5]
torch.gt(a, 0.3) & torch.lt(a, 0.5)
# tensor([[False,  True, False],
#         [ True, False, False]])

# a中位于区间[0.3, 0.5]之间的, 用zero(0)替换,否则a替换,即不变
print(torch.where(torch.gt(a, 0.3) & torch.lt(a, 0.6), zero, a))
# tensor([[0.2620, 0.0000, 0.5924],
#         [0.0000, 0.0475, 0.5491]])

"""


if __name__ == "__main__":
    name = os.listdir(os.path.join("aeroscapes", "SegmentationClass"))

    if not os.path.exists(os.path.join("aeroscapes", "SegmentationClass_road")):
        os.makedirs(os.path.join("aeroscapes", "SegmentationClass_road"))
        
    for index, image_mask_name in enumerate(name):
        image_path = os.path.join("aeroscapes", "SegmentationClass", image_mask_name)
        print(index, image_path)
        to_road(image_path)

    # to_road(r"aeroscapes\SegmentationClass_road\038012_064.png")