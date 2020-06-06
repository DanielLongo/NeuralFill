import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import sys
sys.path.append("../")
sys.path.append("../metrics/")

from metrics_utils import save_checkpoint_metrics
from utils import reduce_channel_batch

def train(epoch, loader, model, optimizer, scheduler, device, writer, log_interval=10, criterion=nn.MSELoss(), save_filename="results_fill/sample"):
	model.train()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	for i, (signals) in enumerate(loader):
		model.zero_grad()

		signals = signals.to(device).view(signals.shape[0], signals.shape[1], -1)
		target_channel = signals[:, 1, :]
		signals_reduced = reduce_channel_batch(signals, 1, a=0) # reduce middle channel to 0
		
		outputs = model(signals_reduced)
		reconstructed = outputs[0] # first output is always the recon 
		recon_loss = criterion(reconstructed, target_channel)
		loss = model.loss_function(target_channel, outputs)
		loss.backward()

		if scheduler is not None:
			scheduler.step()

		optimizer.step()

		mse_sum += recon_loss.item() * signals.shape[0]
		mse_n += signals.shape[0]

		lr = optimizer.param_groups[0]['lr']

		loader.set_description(
			(
				f'TRAIN epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
				f'loss: {loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
				f'lr: {lr:.5f}'
			)
		)

		running_recon_loss += (recon_loss.item() * signals.shape[0]) / signals.shape[0]
		running_loss += (loss.item() * signals.shape[0]) / signals.shape[0]

		if i % log_interval == 0 and i != 0:
			model.eval()

			save_checkpoint_metrics(
				writer=writer,
				model=model,
				sample=signals,
				save_filename=save_filename,
				epoch=epoch,
				iteration=epoch * len(loader) + i,
				loss=running_loss / log_interval, 
				recon_loss=running_recon_loss / log_interval,
				prefix='train',
			)

			running_recon_loss = 0
			running_loss = 0
			model.train()

def eval(epoch, loader, model, device, writer, log_interval=10,  criterion=nn.L1Loss(), save_filename="results_fill/sample"):
	model.eval()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	with torch.no_grad():
		for i, (signals) in enumerate(loader):
			
			signals = signals.to(device).view(signals.shape[0], signals.shape[1], -1)
			target_channel = signals[:, 1, :]
			signals_reduced = reduce_channel_batch(signals, 1, a=0) # reduce middle channel to 0

			outputs = model(signals_reduced)
			reconstructed = outputs[0]
			recon_loss = criterion(reconstructed, target_channel)
			loss = model.loss_function(target_channel, outputs)

			mse_sum += recon_loss.item() * signals.shape[0]
			mse_n += signals.shape[0]

			loader.set_description(
				(
					f'EVAL epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
					f'loss: {loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
				)
			)

			running_recon_loss += (recon_loss.item() * signals.shape[0]) / signals.shape[0]
			running_loss += (loss.item() * signals.shape[0]) / signals.shape[0]


			if i % log_interval == 0:

				save_checkpoint_metrics(
					writer=writer,
					model=model,
					sample=signals,
					save_filename=save_filename,
					epoch=epoch,
					iteration=epoch * len(loader) + i,
					loss=running_loss / log_interval, 
					recon_loss=running_recon_loss / log_interval,
					prefix='eval',
				)

				running_recon_loss = 0
				running_loss = 0