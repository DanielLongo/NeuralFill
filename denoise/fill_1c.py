import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import numpy as np
import time

import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../models/vq-vae-2-pytorch/")
sys.path.append("../data/")
sys.path.append("../metrics/")

from scheduler import CycleScheduler
from metrics_utils import get_metrics, get_recon_metrics
from utils import save_gens_samples, save_reconstruction, loss_function_vanilla, find_valid_filename, check_valid_filename, save_run
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
	model_name = "nn"
	# model_name = "cvq-2-mc"
	# model_name = HOME_PATH + "reconstruction/saved_models/" + "vq-2-mc"

	if check_valid_filename(model_name):
		# the model name is filepath to the model
		saved_model = True
	else:
		saved_model = False

	z_dim = 30
	lr = 1e-3
	# sched = 'cycle'
	sched = None

	num_epochs = 200
	batch_size = 64
	num_examples_train = -1
	num_examples_eval = -1
	
	device = 'cuda'
	normalize = True
	log_interval = 1
	tb_save_loc = "runs/testing/"
	select_channels = [0] #[0,1,2,3]
	num_channels = len(select_channels)

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
		"unet" : UNet(in_channels=num_channels),

		"vq-2-mc" : VQVAE_2(in_channel=num_channels),
		"cvq-2-mc" : cVQVAE_2(in_channel=num_channels),
		"cnn-mc" : ConvVAE(num_channels=num_channels, num_channels_out=num_channels, z_dim=z_dim),

		"avg-interp" : AvgInterpolation(),
		"mne-interp" : MNEInterpolation(),
		"mne-interp-mc" : MNEInterpolationMC(),
	}

	model_filenames = {
		"nn" : HOME_PATH + "models/VAE1c.py",
		"cnn" : HOME_PATH + "models/conv_VAE.py",
		"vq" : HOME_PATH + "models/VQ_VAE_1c.py",
		"vq-2" : HOME_PATH + "models/vq-vae-2-pytorch/vqvae.py",
		"unet" : HOME_PATH + "models/unet.py",
		"cnn-mc" : HOME_PATH + "models/conv_VAE.py",
		"vq-2-mc" : HOME_PATH + "models/vq-vae-2-pytorch/vqvae.py",
		"mne-interp-mc" : HOME_PATH + "denoise/fill_baseline_models.py",
	}



	if saved_model:
		model = torch.load(model_name)
		length = 1024 # TODO find way to set auto
	else:
		model = models[model_name]
		length = lengths[model_name]

	if model_name == "mne-interp" or model_name == "mne-interp-mc":
		select_channels = [0,1,2,3]
	# else:
		# select_channels = [0,1,2]
		# select_channels = [0,1]#,2,3]

	train_files =  TRAIN_NORMAL_FILES_CSV #TRAIN_FILES_CSV 
	# train_files =  TRAIN_FILES_CSV 


	eval_files =  DEV_NORMAL_FILES_CSV #DEV_FILES_CSV 
	# eval_files =  DEV_FILES_CSV 
	eval_dataset = EEGDatasetMc(eval_files, max_num_examples=num_examples_eval, length=length, normalize=normalize, select_channels=select_channels)
	eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
	

	target_filename = model_name
	run_filename = find_valid_filename(target_filename, HOME_PATH + 'denoise/' + tb_save_loc)
	tb_filename = tb_save_loc + run_filename
	writer = SummaryWriter(tb_save_loc + run_filename)

	model = model.to(device)

	try:
		optimizer = optim.Adam(model.parameters(), lr=lr)
		train_model = True
	except ValueError:
		print("This Model Cannot Be Optimized")
		train_model = False
		sched = None

	if train_model:
		train_dataset = EEGDatasetMc(train_files, max_num_examples=num_examples_train, length=length, normalize=normalize, select_channels=select_channels)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		print("m Train Dataset", len(train_dataset))

	print("m Eval Dataset", len(eval_dataset))

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

	save_dir = HOME_PATH + "denoise/saved_runs/" + str(int(time.time())) + "/"
	recon_file = HOME_PATH + "denoise/fill_1c.py"
	train_file = HOME_PATH + "denoise/train_fill_1c.py"
	model_filename = model_filenames[model_name]
	python_files = [recon_file, train_file, model_filename]

	info_dict = {
		"model_name" : model_name,
		"z_dim" : z_dim,
		"lr" : lr,
		"sched" : sched,

		"num_epochs" : num_epochs,
		"batch_size" : batch_size,
		"num_examples_train" : num_examples_train,
		"num_examples_eval" : num_examples_eval,

		"train_files" : train_files,
		"eval_files" : eval_files, 
		
		"device" : device,
		"normalize" : normalize,
		
		"log_interval" : log_interval,
		"tb_dirpath" : tb_filename
	}

	save_run(save_dir, python_files, model, info_dict)

if __name__ == "__main__":
	main()
