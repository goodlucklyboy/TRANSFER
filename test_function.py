import torch
import copy
import time
import mmd
import ResNet
import make_dataloader as D
import torch.nn as nn
import visdom
import torchvision
from torch.autograd import Variable
import torch.optim as optim
root_path = "./data/7_office/"
source_path = "amazon"
target_path = "dslr"
batch_size = 64
epoch_num = 1000
source_data_loader = D.make_data_loader(root_path+source_path)
target_data_loader = D.make_data_loader(root_path+target_path)
len_source_loader = len(source_data_loader)
len_target_loader = len(target_data_loader)
print(len_source_loader,len_target_loader)