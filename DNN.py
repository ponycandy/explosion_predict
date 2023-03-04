import torch
class MyDNN(torch.nn.Module):#这里要求的可变性较小
    def __init__(self):
        super(MyDNN, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=3
        D_in=self.inputshape
        Hidden_1=10
        Hidden_2=10
        D_out=2
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) # 1*200*100的矩阵
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)# 1*200*100的矩阵


    # 定义前向传播
    def forward(self, x):
        #先看actionNet的梯度计算吧
        x = x.to(torch.float32)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)

        return x

    def get_input_shape(self):
        return self.inputshape
    def clean_grad(self):
        self.zero_grad()