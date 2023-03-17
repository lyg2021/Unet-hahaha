from unet import Unet
import torch
import torchvision
import os
from PIL import Image

"""输入模型，预测结果"""

transform = torchvision.transforms.Compose([    
    torchvision.transforms.Resize(size=(256, 512)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean= (0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = Unet().to(DEVICE)

# 导入权重
weight_path = r"useful_weight\unet_30_0_20230310103018.pth"

# 判断是否存在已有的权重文件
if os.path.exists(weight_path):
    
    # 加载权重文件
    state_dict = torch.load(weight_path)
    # print(type(state_dict))

    model.load_state_dict(state_dict=state_dict)
    print("已加载权重文件: {}".format(weight_path))
else:
    print("没有存在的权重文件")

image_path = input("请输入图片路径：")  # aeroscapes\JPEGImages\000001_007.jpg

# 读取图片
image = Image.open(image_path)

# 将图片toTensor归一化至[0, 1]，再normalize到[-1, 1]（有偏差，因为使用的是imagenet的均值和方差）
image_tensor = transform(img=image)

# 升维，让输入符合网络
image_tensor = torch.unsqueeze(input=image_tensor, dim=0)
# print(image_tensor, image_tensor.shape)

# 放到设备上
image_tensor = image_tensor.to(DEVICE)

# 经过网络
output_image_tensor = model(image_tensor)

# 对输出结果进行二值化处理
# output_image_tensor[output_image_tensor >= 0.5] = 1
# output_image_tensor[output_image_tensor < 0.5] = 0

# 将归一化后的灰度图还原，方便观察
output_image_tensor *= 255

#保存输出结果
torchvision.utils.save_image(output_image_tensor, os.path.join("predict_image", "test1.jpg"))
# print(output_image_tensor, output_image_tensor.shape)




