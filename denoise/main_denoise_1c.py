from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../data/")
sys.path.append("../artifacts/")
from utils import save_gens_samples, save_denoise_reconstruction, loss_function_vanilla, custom_norm_batch
from VAE1c import VAE1c
from load_EEGs_1c import EEGDataset1c
from synthetic_artifacts_1c import SyntheticArtifiacts1c 
from blink_1c import Blink1c
from constants import *

torch.manual_seed(1)

batch_size = 128
epochs = 1000
num_examples = 128*4
cuda = torch.cuda.is_available()
log_interval = 10
z_dim = 20
files_csv = ALL_FILES_SEC

dataset = EEGDataset1c(files_csv, max_num_examples=num_examples, length=784)
train_loader = data.DataLoader(
	dataset=dataset,
	shuffle=True,
	batch_size=batch_size,
	pin_memory=cuda
  )

eval_loader = train_loader

# artifacts = SyntheticArtifiacts1c(batch_size*1, length=784)
artifacts = Blink1c(batch_size*1, length=784)

artifacts_loader = data.DataLoader(
	dataset=artifacts,
	shuffle=True,
	batch_size=batch_size,
	pin_memory=cuda
)

artifacts_loader_iter = iter(artifacts_loader)

model = VAE1c(z_dim=z_dim)

if cuda:
	model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
	global artifacts_loader_iter
	model.train()
	train_loss = 0
	for batch_idx, (data) in enumerate(train_loader):
		
		try:
			cur_artifact = next(artifacts_loader_iter)
		except StopIteration:
			artifacts_loader_iter = iter(artifacts_loader)
			cur_artifact = next(artifacts_loader_iter)


		if cuda:
			data, cur_artifact = data.cuda(), cur_artifact.cuda()

		data_noisy = (data + cur_artifact)/2
		# data_noisy = custom_norm_batch(data_noisy)
		
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data_noisy)
		loss = loss_function_vanilla(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				loss.item() / len(data)))

		if batch_idx == 0:
			save_denoise_reconstruction(data.view(-1, 784)[:16], data_noisy.view(-1, 784)[:16], recon_batch[:16], "results_denoise/" + str(epoch) + "_blink_train_recon")

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def eval(epoch):
	global artifacts_loader_iter
	model.eval()
	eval_loss = 0
	with torch.no_grad():
		for batch_idx, (data) in enumerate(eval_loader):

			try:
				cur_artifact = next(artifacts_loader_iter)
			except StopIteration:
				artifacts_loader_iter = iter(artifacts_loader)
				cur_artifact = next(artifacts_loader_iter)


			if cuda:
				data, cur_artifact = data.cuda(), cur_artifact.cuda()

			data_noisy = (data + cur_artifact)/2
			# data_noisy = custom_norm_batch(data_noisy)

			recon_batch, mu, logvar = model(data_noisy)
			eval_loss += loss_function_vanilla(recon_batch, data, mu, logvar).item()
			if batch_idx == 0:
				save_denoise_reconstruction(data.view(-1, 784)[:16], data_noisy.view(-1, 784)[:16], recon_batch[:16], "results_denoise/" + str(epoch) + "_blink_eval_recon")

	eval_loss /= len(eval_loader.dataset)
	print('====> eval set loss: {:.4f}'.format(eval_loss))

if __name__ == "__main__":
	for epoch in range(1, epochs + 1):
		train(epoch)
		eval(epoch)
		save_filename = 'results_denoise/blink_sample_' + str(epoch)
		save_gens_samples(model, save_filename, z_dim, n_samples=16)