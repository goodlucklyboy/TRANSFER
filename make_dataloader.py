#这个文件是用来制作Dataloaderd的，所有的Dataloader制作都写在这里面。Main函数中直接传入参数就行了。
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.datasets import ImageFolder
#传入要训练的图片所在文件夹位置就行。注意区分训练和测试文件夹。返回可以直接用于训练的train_loader,test_loader;
def make_dataloader(train_root,test_root,batch_size=64,shuffle=True,num_workers=0,train_drop_last=False,test_drop_last=False):
	transform = transforms.Compose( [transforms.Grayscale(),
	                                 transforms.Resize( 28 ),
	                                 transforms.CenterCrop( 28 ),
	                                 transforms.ToTensor()] )
	train_dataset = ImageFolder( train_root, transform=transform )
	test_dataset = ImageFolder( test_root, transform=transform )
	train_dataloader = DataLoader( train_dataset, batch_size=batch_size, shuffle=shuffle,
	                               num_workers=num_workers, drop_last=train_drop_last )
	test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=shuffle,
	                              num_workers=num_workers, drop_last=test_drop_last )
	return train_dataloader,test_dataloader
#载入Mnist数据集
def make_Mnist_dataloader(batch_size=64):
	transform = transforms.Compose( [transforms.ToTensor()] )
	train_dataset = datasets.MNIST( root='./data/',
	                                train=True,
	                                transform=transform,
	                                download=True )

	test_dataset = datasets.MNIST( root='./data/',
	                               train=False,
	                               download=True,
	                               transform=transform )

	# Data Loader (Input Pipeline)
	train_loader = DataLoader( dataset=train_dataset,
	                                            batch_size=batch_size,
	                                            shuffle=True )

	test_loader = DataLoader( dataset=test_dataset,
	                                           batch_size=batch_size,
	                                           shuffle=False )
	return train_loader,test_loader
def make_CIFFAR10_dataloader(batch_size=64):
	transform = transforms.Compose([transforms.ToTensor()])
	train_dataset = datasets.CIFAR10(root='./data/',
									 train=True,
									 transform=transform,
									 download=False)
	test_dataset = datasets.CIFAR10(root='./data/',
									train=False,
									transform=transform,download=False)
	train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
	test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
	return train_loader,test_loader
