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
from utils import save_gens_samples, save_reconstruction, loss_function_vanilla, find_valid_filename, check_valid_filename
from load_EEGs_mc import EEGDatasetMc
from constants import *
from train_fill_1c import train, eval


from conv_VAE import ConvVAE
from fill_baseline_models import AvgInterpolation, MNEInterpolation, MNEInterpolationMC
from cVAE1c import cVAE1c
from cvqvae import cVQVAE_2
from unet import UNet
from vqvae import VQVAE_2
from cvqvae import cVQVAE_2
# from VQ_VAE_1c import cVQVAE

def main():
	# model_name = "mne-interp-mc"
	# model_name = "vq-2-mc"
	model_name = "cvq-2-mc"
	# model_name = HOME_PATH + "reconstruction/saved_models/" + "vq-2-mc"
	
	if check_valid_filename(model_name):
		# the model name is filepath to the model
		saved_model = True
	else:
		saved_model = False

	z_dim = 30
	lr = 1e-3
	sched = 'cycle'

	num_epochs = 800
	batch_size = 64
	num_examples_train = -1
	num_examples_eval = -1
	
	device = 'cuda'
	normalize = True
	log_interval = 1
	tb_save_loc = "runs/testing/"

	lengths = {
		# Single Channel Outputs
		"nn" : 784,
		"cnn" : 784,
		"vq" : 784,
		"vq-2" : 1024,
		"unet" : 1024,

		# Multichannel Outputs
		"cnn-mc" : 784,
		"vq-2-mc" : 1024,
		"cvq-2-mc" : 1023,

		# Baselines
		"avg-interp" : 784,
		"mne-interp" : 784,
		"mne-interp-mc" : 1023,
	}

	models = {
		"nn" : cVAE1c(z_dim=z_dim),
		"vq-2" : cVQVAE_2(in_channel=1),
		"unet" : UNet(in_channels=3),

		"vq-2-mc" : VQVAE_2(in_channel=4),
		"cvq-2-mc" : cVQVAE_2(in_channel=4),
		"cnn-mc" : ConvVAE(num_channels=3, num_channels_out=3, z_dim=z_dim),

		"avg-interp" : AvgInterpolation(),
		"mne-interp" : MNEInterpolation(),
		"mne-interp-mc" : MNEInterpolationMC(),
	}


	if saved_model:
		model = torch.load(model_name)
		length = 1024 # TODO find way to set auto
	else:
		model = models[model_name]
		length = lengths[model_name]

	if model_name == "mne-interp" or model_name == "mne-interp-mc":
		select_channels = [0,1,2,3]
	else:
		# select_channels = [0,1,2]
		select_channels = [0,1,2,3]

	train_files =  TRAIN_NORMAL_FILES_CSV #TRAIN_FILES_CSV 
	train_dataset = EEGDatasetMc(train_files, max_num_examples=num_examples_train, length=length, normalize=normalize, select_channels=select_channels)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	eval_files =  DEV_NORMAL_FILES_CSV #DEV_FILES_CSV 
	eval_dataset = EEGDatasetMc(eval_files, max_num_examples=num_examples_eval, length=length, normalize=normalize, select_channels=select_channels)
	eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
	

	print("m Train Dataset", len(train_dataset))
	print("m Eval Dataset", len(eval_dataset))

	target_filename = model_name
	run_filename = find_valid_filename(target_filename, HOME_PATH + 'denoise/' + tb_save_loc)
	writer = SummaryWriter(tb_save_loc + run_filename)

	model = model.to(device)

	try:
		optimizer = optim.Adam(model.parameters(), lr=lr)
		train_model = True
	except ValueError:
		print("This Model Cannot Be Optimized")
		train_model = False
		sched = None

	if saved_model:
		train_model = True

	scheduler = None

	if sched == 'cycle':
		scheduler = CycleScheduler(
			optimizer, lr, n_iter=len(train_loader) * num_epochs, momentum=None
		)
	for i in range(1, num_epochs + 1):
		if train_model:
			train(i, train_loader, model, optimizer, scheduler, device, writer, log_interval=log_interval)
		eval(i, eval_loader, model, device, writer, log_interval=log_interval)

if __name__ == "__main__":
	main()
