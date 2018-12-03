#这个文件用来写训练过程。传入的都是前面制作好的数据和定义好的网络。直接调用传参就行了。
import torch
from torch.autograd import Variable
import time
import copy
def do_source_train_test(model,train_data_loader,test_data_loader,optimizer,criterion,epoch_num=500,save_root=None,vis=None):
	if torch.cuda.is_available():
		model = model.cuda()
	best_model_wts = copy.deepcopy( model.state_dict() )
	best_acc = 0.0
	flag = 0
	for epoch in range( epoch_num):
		since = time.time()
		running_loss = 0
		for img, label in train_data_loader:
			input = Variable( img )
			label = Variable( label )
			if torch.cuda.is_available():
				input = input.cuda()
				label = label.cuda()
			optimizer.zero_grad()
			output = model( input )
			loss = criterion( output, label )
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print( '%d loss:%.3f,time:%.1f' % (epoch + 1, running_loss, time.time() - since) )
		print( 'the %d finished' % (epoch + 1) )
		epoch_correct = 0
		total = 0
		for images, labels in test_data_loader:
			if torch.cuda.is_available():
				images = images.cuda()
				labels = labels.cuda()
			outputs = model( images )
			_, predicted = torch.max( outputs.data, 1 )
			total += labels.size( 0 )
			epoch_correct += (predicted == labels).sum().tolist()  # epoch_acc本身是个tensor,转换为int 才能取小数部分
		epoch_acc = 100 * epoch_correct / total
		print( '正确率：%.3f %%' % (epoch_acc) )
		if vis:
			epoch_acc = torch.Tensor( [epoch_acc] )
			epochs = torch.Tensor( [epoch] )
			running_loss = torch.Tensor( [running_loss] )
			vis.line(X=epochs, Y=epoch_acc, win='acc1', update='append')
			vis.line(X=epochs, Y=running_loss, win='loss1', update='append')
		if epoch_acc > best_acc:
			best_acc = epoch_acc
			flag = epoch
			if save_root:
			# 复制（保存）效果最好的一次状态
				best_model_wts = copy.deepcopy( model.state_dict() )
	print( '最高正确率：%.3f %% %d' % (best_acc, flag) )
	# 将效果最好的一次参数传回Net
	if save_root:
		model.load_state_dict( best_model_wts )
		torch.save( model, save_root )
		print("model have saved finished")
	print( "trained finished" )
	torch.cuda.empty_cache()
	return model,best_acc,flag