import torch
import torchvision
import os
import time
import numpy
import torch.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from data import *

# 模型
from modelLoad import Model_Load


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Val(IMAGE_SIZE: tuple = (256, 256), 
        BATCH_SIZE: int = 4, 
        model_name: str = "unet", 
        model_weight_path: str = "None", 
        save_path: str = "data_result"):

    # 数据加载，DataLoader(data中定义的继承了Dataset类的类，shuffle是否打乱)
    data_loader = DataLoader(MyDataset(root_path=r"aeroscapes",
                                       mask_image_path=r"SegmentationClass_road",
                                       original_image_path=r"JPEGImages",
                                       txt_imageset_path=r"ImageSets",
                                       mode="val",
                                       image_size=IMAGE_SIZE),
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    # 实例化网络模型，并将其加载到设备上
    model = Model_Load(model_name=model_name).to(DEVICE)

    if os.path.exists(model_weight_path):
        # 加载权重文件
        state_dict = torch.load(model_weight_path)
        model.load_state_dict(state_dict=state_dict)
        print("使用权重文件: {} 进行验证".format(model_weight_path))
    else:
        print("没有存在的权重文件")
        return

    # 模型进入验证模式，不启用 Batch Normalization 和 Dropout
    model.eval()

    Accuracy_total = 0.0
    Precision_total = 0.0
    Recall_total = 0.0
    IOU_total = 0.0
    total_num = 0

    # 也不更新梯度，虽然没有使用loss.backward不会进行反向传播，但torch会自己求导，关掉节省显存
    with torch.no_grad():
    # 开始验证val
        for iterations, (image, segment_image) in enumerate(data_loader):

            # 将tensor加载到设备上
            image, segment_image = image.to(DEVICE), segment_image.to(DEVICE)

            # 输出预测值
            output_image = model(image)

            # print(output_image, output_image.shape)
            # print(segment_image, segment_image.shape)

            TP_TN_FP_FN = Calculate_TP_TN_FP_FN(predict=output_image, target=segment_image)

            Accuracy_total += Accuracy_calculate(TP_TN_FP_FN=TP_TN_FP_FN)
            Precision_total += Precision_calculate(TP_TN_FP_FN=TP_TN_FP_FN)
            Recall_total += Recall_calculate(TP_TN_FP_FN=TP_TN_FP_FN)
            IOU_total += IOU_calculate(predict=output_image, target=segment_image, TP_TN_FP_FN=TP_TN_FP_FN)

            total_num = total_num - total_num + iterations + 1

            """接下来就是计算了，然后输出结果，然后保存结果，over"""

            # break

    Accuracy = Accuracy_total / total_num
    Precision = Precision_total / total_num
    Recall = Recall_total / total_num
    IOU = IOU_total / total_num

    with open(os.path.join(save_path, "Accuracy.txt"), mode="a", encoding="utf-8") as file:
        file.write("{}\n".format(Accuracy))

    with open(os.path.join(save_path, "Precision.txt"), mode="a", encoding="utf-8") as file:
        file.write("{}\n".format(Precision))

    with open(os.path.join(save_path, "Recall.txt"), mode="a", encoding="utf-8") as file:
        file.write("{}\n".format(Recall))

    with open(os.path.join(save_path, "IOU.txt"), mode="a", encoding="utf-8") as file:
        file.write("{}\n".format(IOU))

    print(f"Accuracy:{Accuracy}\nPrecision:{Precision}\nRecall:{Recall}\nIOU:{IOU}\n")


def Calculate_TP_TN_FP_FN(predict: torch.Tensor, target: torch.Tensor):
    """计算TP、TN、FP、FN
        输入 两个torch.Tensor
        返回 一个 dict
    """

    # 将 predict 和 target 二值化，方便计算混淆矩阵下的那些值
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    # target 也二值化是因为之前用的transforms.Resize()方法会对标签进行插值，
    # 这样使得原本是二值的标签不是二值了
    target[target >= 0.5] = 1
    target[target < 0.5] = 0

    # 因为都变成了0、1 ，所以直接逐点相乘最后的结果就是相应的值
    # 真就是1， 假就是0，通过torch.where可以把某个tensor中的真假互换
    tp = torch.multiply(predict, target).sum().item()
    tn = torch.multiply(torch.where(predict == 0, 1.0, 0.0),
                        torch.where(target == 0, 1.0, 0.0)).sum().item()
    fp = torch.multiply(predict, torch.where(
        target == 0, 1.0, 0.0)).sum().item()
    fn = torch.multiply(torch.where(
        predict == 0, 1.0, 0.0), target).sum().item()

    # 将计算好的值加入字典，后面要算方便取，最后返回这个字典
    TP_TN_FP_FN = dict()
    TP_TN_FP_FN["tp"] = tp
    TP_TN_FP_FN["tn"] = tn
    TP_TN_FP_FN["fp"] = fp
    TP_TN_FP_FN["fn"] = fn

    return TP_TN_FP_FN


def Accuracy_calculate(TP_TN_FP_FN: dict):
    """准确率(Accuracy),预测结果中正确的占总预测值的比例，它对应语义分割的像素准确率 PA"""
    tp = TP_TN_FP_FN["tp"]
    tn = TP_TN_FP_FN["tn"]
    fp = TP_TN_FP_FN["fp"]
    fn = TP_TN_FP_FN["fn"]

    try:
        Accuracy = (tp + tn) / float(tp + tn + fp + fn)
    except ZeroDivisionError:
        Accuracy = 0.0
    
    return Accuracy

def Precision_calculate(TP_TN_FP_FN: dict):
    """精准率(Precision)又称查准率,预测结果中某类别预测正确的概率，对应语义分割的类别像素准确率 CPA"""
    tp = TP_TN_FP_FN["tp"]
    tn = TP_TN_FP_FN["tn"]
    fp = TP_TN_FP_FN["fp"]
    fn = TP_TN_FP_FN["fn"]

    try:
        Precision = tp / float(tp + fp)
        # Precision2 = tn / float(tn + fn)
    except ZeroDivisionError:
        Precision = 0.0
    
    return Precision

def Recall_calculate(TP_TN_FP_FN: dict):
    """召回率(Recall)又称查全率,预测结果中某类别预测正确的概率,在语义分割常用指标没有对应关系"""
    tp = TP_TN_FP_FN["tp"]
    tn = TP_TN_FP_FN["tn"]
    fp = TP_TN_FP_FN["fp"]
    fn = TP_TN_FP_FN["fn"]

    try:
        Recall = tp / float(tp + fn)
        # Recall2 = tn / float(tn + fp)
    except ZeroDivisionError:
        Recall = 0.0
    
    return Recall

def IOU_calculate(predict:torch.Tensor, target:torch.Tensor, TP_TN_FP_FN: dict):
    """IOU 交并比"""
    tp = TP_TN_FP_FN["tp"]
    tn = TP_TN_FP_FN["tn"]
    fp = TP_TN_FP_FN["fp"]
    fn = TP_TN_FP_FN["fn"]

    try:
        # intersection = torch.multiply(predict, target)
        # union = torch.add(predict, target)
        # IOU = intersection.sum().item() / (union.sum().item() + 1e-10)
        ##################################################  # 这个计算精确，IOU相对会小

        IOU = tp / float(tp + fp + fn)  # 这个计算的时候将置信度高于阈值的都设为正样本，低于阈值的都设为负样本，IOU会高些
    except ZeroDivisionError:
        IOU = 0.0
    
    return IOU


if __name__ == "__main__":
    """ model_name = ['unet', 'setr', 'deeplabv3plus_resnet50', 
    'deeplabv3_resnet50', 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32'] """

    model_name = "deeplabv3plus_hrnetv2_32"

    Val(IMAGE_SIZE=(512, 512),
        BATCH_SIZE=16,
        model_name="deeplabv3_resnet50",
        model_weight_path=r"output/20230318161715/weight/deeplabv3_100_20230318161715.pth"
        )
