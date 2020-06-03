import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


import sys
import numpy as np

sys.path.append("../data/")
sys.path.append("../")
sys.path.append("../models/vq-vae-2-pytorch/")
sys.path.append("../metrics/")

from metrics_utils import get_recon_metrics
from load_EEGs_1c import EEGDataset1c
from utils import find_valid_filename
from constants import *
from vqvae import VQVAE_2
from scheduler import CycleScheduler


def train(epoch, loader, model, optimizer, scheduler, device, writer, log_interval=10):
	model.train()
	loader = tqdm(loader)

	criterion = nn.MSELoss()

	latent_loss_weight = 0.25
	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0

	for i, (img) in enumerate(loader):
		model.zero_grad()

		img = img.to(device).view(img.shape[0], -1)

		out, latent_loss = model(img)
		recon_loss = criterion(out, img)
		latent_loss = latent_loss.mean()
		loss = recon_loss + latent_loss_weight * latent_loss
		loss.backward()

		if scheduler is not None:
			scheduler.step()
		optimizer.step()

		mse_sum += recon_loss.item() * img.shape[0]
		mse_n += img.shape[0]

		lr = optimizer.param_groups[0]['lr']

		loader.set_description(
			(
				f'TRAIN epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
				f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
				f'lr: {lr:.5f}'
			)
		)

		running_loss += (recon_loss.item() * img.shape[0]) / img.shape[0]

		if i % log_interval == 0 and i != 0:
			model.eval()

			iteration = epoch * len(loader) + i

			# hist_diff, spec_diff = get_metrics(model, dataset=loader, z_dim=, n_dataset=100, n_model=100, print_results=False)

			# sample = img[:sample_size]
			sample = img

			with torch.no_grad():
				out, _ = model(sample)
				out, sample = out.view(-1, 1024).cpu().numpy(), sample.view(-1, 1024).cpu().numpy()
				combined = np.vstack((sample[:sample_size], out[:sample_size]))

				np.save("results_recon/train_vq-vae-2_recon_" + str(epoch), combined)

				fft_diff, hist_diff = get_recon_metrics(out, sample)
				writer.add_scalar('train/fft diff', fft_diff, iteration)
				writer.add_scalar('train/hist diff', hist_diff, iteration)
				writer.add_scalar('train/loss', running_loss / log_interval, iteration)

			running_loss = 0
			model.train()

def eval(epoch, loader, model, device, writer, log_interval=10):
	model.eval()

	loader = tqdm(loader)

	criterion = nn.MSELoss()

	latent_loss_weight = 0.25
	sample_size = 16

	mse_sum = 0
	mse_n = 0
	running_loss = 0
	with torch.no_grad():
		for i, (img) in enumerate(loader):
			model.zero_grad()

			img = img.to(device).view(img.shape[0], -1)

			out, latent_loss = model(img)
			recon_loss = criterion(out, img)
			latent_loss = latent_loss.mean()
			loss = recon_loss + latent_loss_weight * latent_loss

			mse_sum += recon_loss.item() * img.shape[0]
			mse_n += img.shape[0]

			lr = optimizer.param_groups[0]['lr']

			loader.set_description(
				(
					f'EVAL epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
					f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
					f'lr: {lr:.5f}'
				)
			)

			running_loss += (recon_loss.item() * img.shape[0]) / img.shape[0]

			if i % log_interval == 0:# and i != 0:

				iteration = epoch * len(loader) + i

				# hist_diff, spec_diff = get_metrics(model, dataset=loader, z_dim=, n_dataset=100, n_model=100, print_results=False)
				# writer.add_scalar('train/hist diff', hist_diff, iteration)
				# writer.add_scalar('train/spec diff', spec_diff, iteration)

				if i != 0:
					running_loss /= log_interval # take average 


				# sample = img[:sample_size]
				sample = img

				out, _ = model(sample)
				out, sample = out.view(-1, 1024).cpu().numpy(), sample.view(-1, 1024).cpu().numpy()
				combined = np.vstack((sample[:sample_size], out[:sample_size]))

				np.save("results_recon/eval_vq-vae-2_recon_" + str(epoch), combined)

				fft_diff, hist_diff = get_recon_metrics(out, sample)
				writer.add_scalar('eval/fft diff', fft_diff, iteration)
				writer.add_scalar('eval/hist diff', hist_diff, iteration)
				writer.add_scalar('eval/loss', running_loss / log_interval, iteration)

				running_loss = 0



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size', type=int, default=256)
	# parser.add_argument('--epoch', type=int, default=560)
	parser.add_argument('--epoch', type=int, default=1000)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--sched', type=str)

	args = parser.parse_args()

	print(args)

	device = 'cuda'

	train_files =  TRAIN_NORMAL_FILES_CSV #TRAIN_FILES_CSV 
	train_dataset = EEGDataset1c(train_files, max_num_examples=-1, length=(32*32))
	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

	eval_files =  DEV_NORMAL_FILES_CSV #DEV_FILES_CSV 
	eval_dataset = EEGDataset1c(eval_files, max_num_examples=-1, length=(32*32))
	eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True, num_workers=4)
	

	print("len train dataset", len(train_dataset))
	print("len eval dataset", len(eval_dataset))

	target_filename = "vq-vae-2"
	run_filename = find_valid_filename(target_filename, HOME_PATH + 'reconstruction/runs/')
	writer = SummaryWriter('runs/' + run_filename)

	model = VQVAE_2(in_channel=1).to(device)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = None

	if args.sched == 'cycle':
		scheduler = CycleScheduler(
			optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
		)

	for i in range(1, args.epoch + 1):
		train(i, train_loader, model, optimizer, scheduler, device, writer, log_interval=3)
		eval(i, eval_loader, model, device, writer, log_interval=3)
		# torch.save(model.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt')
