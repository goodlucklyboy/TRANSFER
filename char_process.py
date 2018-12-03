import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import time
from torch.autograd import Variable
import visdom
vis = visdom.Visdom(env=u'acc1')

batch_size = 64
Test_batch_size = 64
all_root = 'F:/EnglishHnd/English/Hnd/Img/'
num_train_root = 'F:/EnglishHnd/num/train/'
num_test_root = 'F:/EnglishHnd/num/test/'
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(28),
                                transforms.CenterCrop(28),
                                transforms.ToTensor() ])
train_dataset = ImageFolder(num_train_root,transform = transform)
test_dataset = ImageFolder(num_test_root,transform=transform)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False)
#to_img = transforms.ToPILImage()
#img = to_img(dataset[500][0])
#img.show()
class Lenet5( nn.Module ):
	def __init__(self):
		super( Lenet5, self ).__init__()

		self.Conv = nn.Sequential( nn.Conv2d( 1, 6, 5, stride=1, padding=1 ),
		                     nn.MaxPool2d(2,2),
							nn.Conv2d( 6, 16, 5, stride=1, padding=0 ),
							nn.MaxPool2d( 2, 2 ))

		self.Fc = nn.Sequential( nn.Linear( 256, 120 ),
		                   nn.Linear( 120, 84 ),
		                   nn.Linear( 84, 10 ))

	def forward(self, x):
		out = self.Conv(x)
		out = out.view(out.size(0),-1)
		out = self.Fc(out)
		return out

net_name = 'first_time.pkl'
Transfer_net = torch.load(net_name)
for param in Transfer_net.parameters():
	param.requires_grad = False
Transfer_net.Fc = nn.Sequential( nn.Linear( 256, 120 ),
		                   nn.Linear( 120, 84 ),
		                   nn.Linear( 84, 10 ))
Transfer_net = Transfer_net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( Transfer_net.Fc.parameters(), lr=0.001, momentum=0.9 )
best_acc = 0.0
for epoch in range(500):
	since = time.time()
	running_loss = 0
	for img, label in train_dataloader:
		input = Variable( img ).cuda()
		label = Variable( label ).cuda()
		optimizer.zero_grad()
		output = Transfer_net( input )
		loss = criterion( output, label )
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	print( '%d loss:%.3f,time:%.1f' % (epoch + 1, running_loss, time.time() - since) )
	print( 'the %d finished' % (epoch + 1) )
	epoch_correct = 0
	total = 0
	for images, labels in test_dataloader:
		inputs = Variable( images ).cuda()
		labels = Variable( labels ).cuda()
		outputs = Transfer_net( inputs )
		_, predicted = torch.max( outputs.data, 1 )
		total += labels.size( 0 )
		epoch_correct += (predicted == labels).sum().tolist()  # epoch_acc本身是个tensor,转换为int 才能取小数部分
	epoch_acc = 100 * epoch_correct / total
	epoch_correct = torch.Tensor( [epoch_correct] )
	epoch_acc = torch.Tensor( [epoch_acc] )
	epochs = torch.Tensor( [epoch] )
	running_loss = torch.Tensor( [running_loss] )
	vis.line( X=epochs, Y=epoch_acc, win='acc', update='append' )
	vis.line(X=epochs,Y=running_loss,win='loss',update='append')
	print( '正确率：%.3f %%' % (epoch_acc) )
	if epoch_acc > best_acc:
		best_acc = epoch_acc
		# 复制（保存）效果最好的一次状态
print( '最高正确率：%.3f %%' % (best_acc))

print("finish")


