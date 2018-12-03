from make_dataloader import make_CIFFAR10_dataloader
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
classes = ('plane','car','bird','cat','deer','dog','frog','horsr','ship','truck')
train_loader,test_loader=make_CIFFAR10_dataloader(4)
dataiter = iter(train_loader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(labels)
print(''.join('%s'%classes[labels[j]] for j in range(4)))
print(classes[1])
print(len(train_loader),len(test_loader))
print('finished')