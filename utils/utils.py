from PIL import Image

def keep_image_size_open_Image(path, size=(512, 512)):
    """对图片进行预处理"""

    # 获取图片
    image = Image.open(path)

    # 获取最长边的尺寸
    temp = max(image.size)
    # print(temp)

    # 做一个 mask 掩码
    mask = Image.new(mode="RGB", size=(temp, temp), color=(0, 0, 0))

    # 将原图粘贴上去，(0, 0) 是 image 距离 mask 左上角的距离
    mask.paste(image, (0, 0))
    # mask.show()

    # 调整大小
    mask = mask.resize(size=size)
    # mask.show()

    mask.resize(size=size)

    return mask

def keep_image_size_open_Segment(path, size=(512, 512)):
    """对图片进行预处理"""

    # 获取图片
    image = Image.open(path)

    # 获取最长边的尺寸
    temp = max(image.size)
    # print(temp)

    # 做一个 mask 掩码
    mask = Image.new(mode="F", size=(temp, temp), color=0)

    # 将原图粘贴上去，(0, 0) 是 image 距离 mask 左上角的距离
    mask.paste(image, (0, 0))
    # mask.show()

    # 调整大小
    mask = mask.resize(size=size)
    # mask.show()

    mask.resize(size=size)

    return mask


if __name__ == "__main__":
    # path = r"aeroscapes/JPEGImages/000001_001.jpg"
    # keep_image_size_open(path=path)
    pass