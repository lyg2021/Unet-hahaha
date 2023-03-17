import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_block(nn.Module):
    """卷积层,(Conv3*3, Norm, 激活)*2"""
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.layer = nn.Sequential(

            # 'reflect'是以矩阵边缘为对称轴，将矩阵中的元素对称的填充到最外围.
            # 3*3卷积，步长1，padding=1
            # 卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            
            nn.BatchNorm2d(out_channel),

            # dropout舍弃一些,tensor中部分值（百分之30）置零
            nn.Dropout(0.3),

            # 激活
            nn.LeakyReLU(),

            # 重复
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    """下采样层，和原文的maxpool不同，通过步长为2的3*3卷积进行下采样"""
    def __init__(self, in_channel) -> None:
        # 下采样不改变通道数，in_channel = out_channel
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    """上采样，使用最邻近插值法"""
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            # 使用1*1卷积将通道数减半，上采样后会和之前的相近尺寸的特征图进行拼接
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=1, stride=1)
        )

    def forward(self, x):

        # F.interpolate——数组采样操作, 
        # scale_factor(float或序列)：空间大小的乘数, 
        # mode(str)：用于采样的算法
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.layer(x)
        return x


class Unet(nn.Module):
    """Unet模块组装"""
    def __init__(self) -> None:
        super().__init__()
        # 卷积、下采样轮着来
        self.c1 = Conv_block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_block(512, 1024)      

        # 上采样、卷积轮着来
        # 上采样后通道数会减半，但会和前面的特征图拼接
        self.u1 = UpSample(1024)
        self.c6 = Conv_block(1024, 512)
        self.u2 = UpSample(512)   
        self.c7 = Conv_block(512, 256)
        self.u3 = UpSample(256)   
        self.c8 = Conv_block(256, 128)
        self.u4 = UpSample(128)   
        self.c9 = Conv_block(128, 64)

        # 输出
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        self.Th = nn.Sigmoid()
        # self.Th = nn.Softmax(dim=0)


    def forward(self, x):
        # Unet 左边部分,每个Left_Feature是卷积后的特征图，第一个只有卷积，后面都是先下采样后卷积
        Left_Feature1 = self.c1(x)
        Left_Feature2 = self.c2(self.d1(Left_Feature1))
        Left_Feature3 = self.c3(self.d2(Left_Feature2))
        Left_Feature4 = self.c4(self.d3(Left_Feature3))
        Left_Feature5 = self.c5(self.d4(Left_Feature4))

        # Unet 右边部分，需要对特征图进行拼接，这里的每个Right_Feature是上采样并拼接再卷积后的特征图
        Right_Feature1 = self.c6(torch.cat((self.u1(Left_Feature5), Left_Feature4), dim=1)) 
        Right_Feature2 = self.c7(torch.cat((self.u2(Right_Feature1), Left_Feature3), dim=1)) 
        Right_Feature3 = self.c8(torch.cat((self.u3(Right_Feature2), Left_Feature2), dim=1)) 
        Right_Feature4 = self.c9(torch.cat((self.u4(Right_Feature3), Left_Feature1), dim=1)) 

        return self.Th(self.out(Right_Feature4))
 




if __name__ == "__main__":
    # model = torch.randn(3,3,3)
    # print(model)

    # dropout = nn.Dropout(0.3)
    # model = dropout(model)
    # print(model)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE,type(DEVICE))


    # 测试模型是否有错误，顺便测试下GPU的性能够不够，6G的显存，512*512的大小，batch_size=2 就直接爆显存了
    x = torch.randn(4, 3, 256, 256)
    x = x.to(DEVICE)
    net = Unet().to(DEVICE)
    print(net(x).shape)
    pass

