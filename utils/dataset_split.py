# -*- coding: utf-8 -*-
"""
Author:smile
Date:2022/09/11 10：00
顺序：脚本A1
简介：分训练集、验证集和测试集，按照 8：1：1 的比例来分，训练集8，验证集1，测试集1
     
"""
import os
import random
import argparse

parser = argparse.ArgumentParser()
# mask文件的地址，根据自己的数据进行修改
parser.add_argument('--mask_path', default=r'aeroscapes\SegmentationClass_road', type=str, help='input segmentation mask path')
# 数据集的划分
parser.add_argument('--txt_path', default=r'aeroscapes\ImageSets', type=str, help='output txt label path')
opt = parser.parse_args()

train_percent = 0.6  # 训练集所占比例
val_percent = 0.2    # 验证集所占比例
test_persent = 0.2   # 测试集所占比例

maskfilepath = opt.mask_path
txtsavepath = opt.txt_path
total_mask = os.listdir(maskfilepath)

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_mask)  
list = list(range(num))

t_train = int(num * train_percent)  
t_val = int(num * val_percent)

train = random.sample(list, t_train)
num1 = len(train)
for i in range(num1):
    list.remove(train[i])


val_test = [i for i in list if not i in train]
val = random.sample(val_test, t_val)
num2 = len(val)
for i in range(num2):
    list.remove(val[i])


file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')

for i in train:
    name = total_mask[i][:-4] + '\n'
    file_train.write(name)

for i in val:
    name = total_mask[i][:-4] + '\n'
    file_val.write(name)    

for i in list:
    name = total_mask[i][:-4] + '\n'
    file_test.write(name)
    
    
file_train.close()
file_val.close()
file_test.close()

