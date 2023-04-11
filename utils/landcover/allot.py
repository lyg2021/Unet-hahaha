# ├── data
# │   ├── my_dataset
# │   │   ├── img_dir
# │   │   │   ├── train
# │   │   │   │   ├── xxx{img_suffix}
# │   │   │   │   ├── yyy{img_suffix}
# │   │   │   │   ├── zzz{img_suffix}
# │   │   │   ├── val
# │   │   ├── ann_dir
# │   │   │   ├── train
# │   │   │   │   ├── xxx{seg_map_suffix}
# │   │   │   │   ├── yyy{seg_map_suffix}
# │   │   │   │   ├── zzz{seg_map_suffix}
# │   │   │   ├── val

import os
import shutil
import random


def copy_to_new(old_path: str, new_path: str):
    """脱裤子放屁"""
    shutil.copyfile(old_path, new_path)


input_dir = r"output_new"
dataset_root = r"landcover"

train_img_dir = os.path.join(dataset_root, "img_dir", "train")
train_ann_dir = os.path.join(dataset_root, "ann_dir", "train")

val_img_dir = os.path.join(dataset_root, "img_dir", "val")
val_ann_dir = os.path.join(dataset_root, "ann_dir", "val")

test_img_dir = os.path.join(dataset_root, "img_dir", "test")
test_ann_dir = os.path.join(dataset_root, "ann_dir", "test")


if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)

if not os.path.exists(train_ann_dir):
    os.makedirs(train_ann_dir)

if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)

if not os.path.exists(val_ann_dir):
    os.makedirs(val_ann_dir)

if not os.path.exists(test_img_dir):
    os.makedirs(test_img_dir)

if not os.path.exists(test_ann_dir):
    os.makedirs(test_ann_dir)

print("Making directories successfully")


def get_name_lists(txt_path: str):
    """
        返回两个list, 原图名称列表, 标签名称列表
    """
    # 获取对应train.txt或val.txt或test.txt里面的名称列表，顺便加个后缀名比如.jpg
    with open(txt_path,
            mode="r",
            encoding="utf-8") as txt_file:   # 打开这个txt文件，开始读内容

        # 读所有行，所有行共同构成一个列表list
        txt_lines = txt_file.readlines()

        # 训练原图名称列表
        image_name_list = []
        
        # 训练标签名称列表
        mask_name_list = []

        # 把读到的换行符搞掉，加上后缀名
        for index, line in enumerate(txt_lines):
            image_name = line.replace("\n", "") + ".jpg"
            image_name_list.append(image_name)

            mask_name = line.replace("\n", "") + "_m.png"
            mask_name_list.append(mask_name)
        
        return image_name_list, mask_name_list

train_image_name_list, train_mask_name_list = get_name_lists("./train.txt")
val_image_name_list, val_mask_name_list = get_name_lists("./val.txt")
test_image_name_list, test_mask_name_list = get_name_lists("./test.txt")

def copy_to_dir_new_true_true(image_or_mask_list: list, new_dir: str):
    """
        输入:名称列表, 新路径
    """
    for index, file_name in enumerate(image_or_mask_list):        
        copy_to_new(old_path=os.path.join(input_dir, file_name),
                    new_path=os.path.join(new_dir, file_name.replace("_m", "")))
        print(index, "copy", file_name, "to", new_dir)

# 训练原图
copy_to_dir_new_true_true(train_image_name_list, train_img_dir)

# 训练标签
copy_to_dir_new_true_true(train_mask_name_list, train_ann_dir)

# 验证原图
copy_to_dir_new_true_true(val_image_name_list, val_img_dir)

# 验证标签
copy_to_dir_new_true_true(val_mask_name_list, val_ann_dir)

# 测试原图
copy_to_dir_new_true_true(test_image_name_list, test_img_dir)

# 测试标签
copy_to_dir_new_true_true(test_mask_name_list, test_ann_dir)


print("Complete!!!")