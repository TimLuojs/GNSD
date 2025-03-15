from tqdm import tqdm
from model import *
from const import *
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from nsloss import nsloss

def inverse_normalize(data, min_a, max_a):
    min_a = min_a.view(1, -1)
    max_a = max_a.view(1, -1)
    denormalized_data = data * (max_a - min_a) + min_a
    return denormalized_data

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    min_a = np.load(os.path.join('./data', 'min_a.npy'))  
    max_a = np.load(os.path.join('./data', 'max_a.npy')) 
    min_a = torch.tensor(min_a)
    max_a = torch.tensor(max_a)
    feats = dataO.shape[1]
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
    bs = model.batch
    dataloader = DataLoader(dataset, batch_size = bs)
    n = epoch + 1
    l1s = []
    if training:
        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            z = model(window, elem)
            if epoch > 13: # last two round adversarial
                model.unfreeze_layers_decoder('decoder1')
                l1 = -(1 / n) *l(z[1], elem) 
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                model.unfreeze_layers_decoder('decoder2')
                l2 = (1 - 1/n) * l(z[1], elem) + (nsconst) *( nsloss(inverse_normalize(z[0],min_a,max_a)) + nsloss(inverse_normalize(z[1],min_a,max_a)) )
                loss2 = torch.mean(l2)
                loss2.backward()
                optimizer.step()
                model.unfreeze_all()
            else:
                l1 = (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem) + (nsconst) *( nsloss(inverse_normalize(z[0],min_a,max_a)) + nsloss(inverse_normalize(z[1],min_a,max_a)) )
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        scheduler.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        with torch.no_grad():  
            lsnp = []
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)  
                l1 = ( 1/ n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem) + (nsconst) *( nsloss(inverse_normalize(z[0],min_a,max_a)) + nsloss(inverse_normalize(z[1],min_a,max_a)) )
                lsnp.append(l1.detach().cpu().numpy().squeeze())
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
            mean_l1 = np.mean(l1s)
            tqdm.write(f'Test Epoch {epoch},\tL1 = {mean_l1}')
            combined_array = np.vstack(lsnp)
            print(combined_array.shape)
            return combined_array, mean_l1


def load_model(dims):
    if modelname == "GNSDSingle":
        model = GNSDSingleStation(dims).double()
    else:
        model = GNSD(dims).double()
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def load_dataset():
	loader = []
	for file in ['train', 'test', 'labels']:
		loader.append(np.load(f'data/{file}.npy'))
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, _ in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	file_path = f'result.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
 
