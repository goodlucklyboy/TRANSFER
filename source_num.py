# 此程序为手写数字识别，作为迁移学习的源领域之一。完成后会保存识别效果最好的一次状态，并返回到Net中
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import copy
import visdom

# 一个bitack
batch_size = 64
vis = visdom.Visdom(env=u'acc')


# define the struct of the network
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


# MNIST Dataset
transform = transforms.Compose([
								transforms.ToTensor()])
train_dataset = datasets.MNIST( root='./data/',
                                train=True,
                                transform=transform,
                                download=True )

test_dataset = datasets.MNIST( root='./data/',
                               train=False,
                               download=True,
                               transform=transform )

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True )

test_loader = torch.utils.data.DataLoader( dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False )

Net = Lenet5().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( Net.parameters(), lr=0.001, momentum=0.9 )
best_model_wts = copy.deepcopy( Net.state_dict() )
best_acc = 0.0
for epoch in range(10):
	since = time.time()
	running_loss = 0
	for img, label in train_loader:
		input = Variable( img ).cuda()
		label = Variable( label ).cuda()
		optimizer.zero_grad()
		output = Net( input )
		loss = criterion( output, label )
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	print( '%d loss:%.3f,time:%.1f' % (epoch + 1, running_loss, time.time() - since) )
	print( 'the %d finished' % (epoch + 1) )
	epoch_correct = 0
	total = 0
	for images, labels in test_loader:
		inputs = Variable( images ).cuda()
		labels = Variable( labels ).cuda()
		outputs = Net( inputs )
		_, predicted = torch.max( outputs.data, 1 )
		total += labels.size( 0 )
		epoch_correct += (predicted == labels).sum().tolist()  # epoch_acc本身是个tensor,转换为int 才能取小数部分
	epoch_acc = 100*epoch_correct/total
	print( '正确率：%.3f %%' % (epoch_acc ))
	epoch_correct=torch.Tensor([epoch_correct])
	epoch_acc=torch.Tensor([epoch_acc])
	epochs = torch.Tensor([epoch])
	vis.line(X=epochs,Y=epoch_acc,win='acc',update='append')
	vis.line(X=epochs,Y=epoch_correct,win='total',update='append')
	if epoch_acc > best_acc:
		best_acc = epoch_acc
		flag = epoch
		# 复制（保存）效果最好的一次状态
		best_model_wts = copy.deepcopy( Net.state_dict() )
print( '最高正确率：%.3f %% %d' % (best_acc,flag) )
# 将效果最好的一次参数传回Net
Net.load_state_dict( best_model_wts )
torch.save(Net,'firtest_time.pkl')
print("finish")
torch.cuda.empty_cache()

