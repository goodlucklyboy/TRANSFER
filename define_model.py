#这个文件专门用来存储定义的模型，相当于一个头文件.千万不要在里面写类和函数之外的东西。
#其他文件调用的时候，会将这里面的内容写入内存。若果有写一些其他的（如print,会打印出来）。
import torch.nn as nn
#最简单的卷积网络。两个卷积池化层，三层全连接网络。
class Lenet5( nn.Module ):
	def __init__(self):
		super( Lenet5, self ).__init__()

		self.Conv = nn.Sequential( nn.Conv2d( 3, 6, 5, stride=1, padding=1 ),
		                     nn.MaxPool2d(2,2),
							nn.Conv2d( 6, 16, 5, stride=1, padding=0 ),
							nn.MaxPool2d( 2, 2 ))

		self.Fc = nn.Sequential( nn.Linear( 400, 120 ),
		                   nn.Linear( 120, 84 ),
		                   nn.Linear( 84, 26))

	def forward(self, x):
		out = self.Conv(x)
		out = out.view(out.size(0),-1)
		out = self.Fc(out)
		return out


class My_model(nn.Module):
    def __init__(self):
        super(My_model,self).__init__()
        self.Conv = nn.Sequential(nn.Conv2d(3,6,3,stride=1,padding=1),
							    nn.MaxPool2d(2,2),
								nn.ReLU(),
								nn.Conv2d(6,30,3,stride=1,padding=1),
								nn.MaxPool2d(2,2),
								nn.ReLU(),
								nn.Conv2d(30,64,3,stride=1,padding=1),
								nn.MaxPool2d(2,2),
							    nn.ReLU())
        self.Fc = nn.Sequential(nn.Linear(1024,256),
								nn.Linear(256,64),
								nn.Linear(64,10))

    def forward(self, x):
        out = self.Conv(x)
        out = out.view(out.size(0),-1)
        out = self.Fc(out)
        return out

