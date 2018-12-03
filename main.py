#主函数，调用前面的模块的各种函数就行。
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from define_model import My_model
import torchvision
from train_and_eval import do_source_train_test
from make_dataloader import make_CIFFAR10_dataloader
#num_train_root = 'F:/EnglishHnd/num/train/'#手写数字训练集的目录
#num_test_root = 'F:/EnglishHnd/num/test/'
#alp_train_root = 'F:/EnglishHnd/English/train/'
#alp_test_root = 'F:/EnglishHnd/English/test/'
#Pnum_train_root = 'F:/print_num/train/'
#Pnum_test_root = 'F:/print_num/test/'#以上的目录保存在这里，估计也用不着了
batch_size = 64 #每个batch的数量
epoch_num = 1000
trans = My_model()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD( trans.parameters(), lr=0.001, momentum=0.9,weight_decay=0.01 )
vis_env = visdom.Visdom(env=u'是否过拟合，加L2正则化')
train_data_loader,test_data_loader = make_CIFFAR10_dataloader(batch_size)
do_source_train_test(trans,train_data_loader,test_data_loader,optimizer,criterion,epoch_num,save_root=None,vis=vis_env)