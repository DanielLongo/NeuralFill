import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import numpy as np

import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../models/vq-vae-2-pytorch/")
sys.path.append("../data/")
sys.path.append("../metrics/")

from scheduler import CycleScheduler
from metrics_utils import get_metrics, get_recon_metrics
from utils import save_gens_samples, save_reconstruction, loss_function_vanilla, find_valid_filename
from load_EEGs_1c import EEGDataset1c
from constants import *
from train import train, eval


from conv_VAE import ConvVAE
from VAE1c import VAE1c
from vqvae import VQVAE_2
from VQ_VAE_1c import VQVAE

def main():
	model_name = "nn"
	z_dim = 30
	lr = 1e-3
	sched = None

	num_epochs = 1000
	batch_size = 64
	num_examples_train = -1
	num_examples_eval = -1
	
	device = 'cuda'
	normalize = True
	
	log_interval = 2
	tb_save_loc = "runs/compare/"

	lengths = {
		"nn" : 784,
		"cnn" : 784,
		"vq" : 784,
		"vq-2" : 1024,
	}

	models = {
		"nn" : VAE1c(z_dim=z_dim),
		"cnn" : ConvVAE(num_channels=1, z_dim=z_dim),
		"vq" : VQVAE(hidden=z_dim),
		"vq-2" : VQVAE_2(in_channel=1),
	}


	length = lengths[model_name]
	model = models[model_name]

	train_files =  TRAIN_NORMAL_FILES_CSV #TRAIN_FILES_CSV 
	train_dataset = EEGDataset1c(train_files, max_num_examples=num_examples_train, length=length, normalize=normalize)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	eval_files =  DEV_NORMAL_FILES_CSV #DEV_FILES_CSV 
	eval_dataset = EEGDataset1c(eval_files, max_num_examples=num_examples_eval, length=length, normalize=normalize)
	eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
	

	print("m Train Dataset", len(train_dataset))
	print("m Eval Dataset", len(eval_dataset))

	target_filename = model_name
	run_filename = find_valid_filename(target_filename, HOME_PATH + 'reconstruction/' + tb_save_loc)
	writer = SummaryWriter(tb_save_loc + run_filename)

	model = model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr)
	scheduler = None

	if sched == 'cycle':
		scheduler = CycleScheduler(
			optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
		)
	for i in range(1, num_epochs + 1):
		train(i, train_loader, model, optimizer, scheduler, device, writer, log_interval=log_interval)
		eval(i, eval_loader, model, device, writer, log_interval=log_interval)

if __name__ == "__main__":
	main()