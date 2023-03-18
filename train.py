import torch
import torchvision
import os
import time

import torch.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from data import *

# 模型
from modelLoad import Model_Load

from val import Val


# ---超参数---
START_EPOCHS = 1
END_EPOCHS = 100

BATCH_SIZE = 4
IMAGE_SIZE = (512, 512)

BATCH_SIZE_val = 16
IMAGE_SIZE_val = (512, 512)

save_iterations = 200    # 每几个iteration保存一可视化效果图
save_epochs = 5          # 每几个epoch保存一次权重并验证

model_name = "deeplabv3_hrnetv2_32"      # 模型的名称, 用于选择模型
""" model_name = ['unet', 'setr', 'deeplabv3plus_resnet50', 
    'deeplabv3_resnet50', 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32'] """

# ---设备配置---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cup")    # 用于测试，有些在gpu上运行发生的错误不显示，放在cpu上就好了-------------------------很管用！

weight_path = r"none"


# ---路径配置---

# 数据集路径
root_path=r"aeroscapes"                   # 数据集根目录（相对项目目录）
mask_image_path=r"SegmentationClass_road"   # mask目录（相对根目录）
original_image_path=r"JPEGImages"           # 原图目录（相对根目录）
txt_imageset_path=r"ImageSets"              # 划分的txt目录（相对根目录）

# 获取现在的时间
current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 创建输出根目录
output_root_dir = os.path.join("output", "{}".format(current_time))
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)

# 权重保存位置
weight_dir = os.path.join("{}".format(output_root_dir), "weight")
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

# 预测结果可视化对比图片保存位置
view_image_dir = os.path.join("{}".format(output_root_dir), "view_image")
if not os.path.exists(view_image_dir):
    os.makedirs(view_image_dir)

# 实验数据保存位置
data_result_dir = os.path.join("{}".format(output_root_dir), "data_result")
if not os.path.exists(data_result_dir):
    os.makedirs(data_result_dir)


if __name__ == "__main__":

    print("torch version:", torch.__version__)
    print("cuda available(true or false):",
          torch.cuda.is_available())  # cuda是否可用
    print("GPU count:", torch.cuda.device_count())  # 返回GPU的数量
    print("GPU name:", torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始

    # 数据加载，DataLoader(data中定义的继承了Dataset类的类，shuffle是否打乱)
    data_loader = DataLoader(MyDataset(root_path=root_path,
                                       mask_image_path=mask_image_path,
                                       original_image_path=original_image_path,
                                       txt_imageset_path=txt_imageset_path,
                                       mode="train",
                                       image_size=IMAGE_SIZE),
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    # 实例化网络模型，并将其加载到设备上
    model = Model_Load(model_name=model_name)
    model = model.to(DEVICE)

    # 判断是否存在已有的权重文件
    if os.path.exists(weight_path):

        # 加载权重文件
        state_dict = torch.load(weight_path)
        # print(type(state_dict))
        model.load_state_dict(state_dict=state_dict)
        print("已加载权重文件: {}".format(weight_path))

    else:
        print("没有存在的权重文件")

    # 使用Adam优化器优化训练参数
    opt = optim.Adam(model.parameters())

    # 定义损失函数，二元交叉熵 Binary Cross Entropy
    bce = nn.BCELoss()

    ce = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(START_EPOCHS, END_EPOCHS+1):

        # 打开模型的训练模式（可能默认是训练模式，但后面有代码要进入验证模式，保证每次循环前打开它）
        model.train()

        # 损失总和
        train_loss_total = 0.0

        # 获取每个batch的图片和标签图片的tensor（data_loader返回值详见data.py）
        # 一个epoch有【（图片总数）/（batch_size）】个batch
        for iterations, (image, segment_image) in enumerate(data_loader):

            # 将tensor加载到设备上
            image, segment_image = image.to(DEVICE), segment_image.to(DEVICE)

            # 一轮训练
            output_image = model(image)

            # 梯度初始化为0
            opt.zero_grad()

            # 计算损失
            # train_loss = F.binary_cross_entropy(input=output_image, target=segment_image)
            train_loss = bce(output_image, segment_image)
            train_loss_total += train_loss.item()
            train_loss_average = train_loss_total / (iterations+1)

            # 反向传播
            train_loss.backward()

            # 更新梯度
            opt.step()

            # 每5个iteration输出一次损失
            if iterations % 5 == 0:
                print("epoch: {}\t iters: {}\t train_loss: {}".format(
                    epoch, iterations, train_loss_average))
                with open(os.path.join(data_result_dir, "loss.txt"), mode="a", encoding="utf-8") as loss_file:
                    loss_file.write("{:4f}\n".format(train_loss_average))

            # 每xxx次iteration，保存一次训练可视化效果图片
            if iterations % save_iterations == 0:

                # 可视化训练效果
                # 分别拿到原图，标签图，模型输出图
                # image.shape = (batch_size, 3, 256, 256)
                # image[0].shape = (3, 256, 256)
                _image = image[0]
                _segment_image = segment_image[0]
                _output_image = output_image[0]

                # 对输出结果进行二值化处理
                _output_image[_output_image >= 0.5] = 1
                _output_image[_output_image < 0.5] = 0

                # 将归一化后的灰度图还原，方便观察
                _segment_image *= 255
                _output_image *= 255

                # 将单通道图片的第一个维度复制三次，后面两维度保持不变，以便于与三通道的原图拼接
                _segment_image, _output_image = _segment_image.repeat(3, 1, 1), _output_image.repeat(3, 1, 1)

                # 将三张图拼接起来对比看效果
                view_image = torch.stack([_image, _segment_image, _output_image])

                # 保存可视化效果图片
                torchvision.utils.save_image(view_image, os.path.join(
                    view_image_dir, "{}_{}_{}_{}.jpg".format(model_name, epoch, iterations, current_time)))
                
            
        # 每 x 个 Epoch 保存一个权重文件,并进行一次Val
        if epoch % save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(
                weight_dir, "{}_{}_{}.pth".format(model_name, epoch, current_time)))
            
            # 验证
            Val(IMAGE_SIZE=IMAGE_SIZE_val, 
                BATCH_SIZE=BATCH_SIZE_val,
                model_name=model_name,
                model_weight_path=os.path.join(
                weight_dir, "{}_{}_{}.pth".format(model_name, epoch, current_time)),
                save_path=data_result_dir)
