import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from metrics_utils import save_checkpoint_metrics

def train(epoch, loader, model, optimizer, scheduler, device, writer, log_interval=10, criterion=nn.MSELoss(reduction='mean'), save_filename="results_recon/sample"):
	model.train()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	for i, (signals) in enumerate(loader):
		model.zero_grad()

		signals = torch.squeeze(signals.to(device)) #.view(signals.shape[0], -1)

		outputs = model(signals)
		reconstructed = outputs[0]
		recon_loss = criterion(reconstructed, signals)
		loss = model.loss_function(signals, outputs)
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

		running_recon_loss += recon_loss.item()
		running_loss += loss.item()

		if i % log_interval == 0:
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

def eval(epoch, loader, model, device, writer, log_interval=10,  criterion=nn.MSELoss(reduction='mean'), save_filename="results_recon/sample"):
	model.eval()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	with torch.no_grad():
		for i, (signals) in enumerate(loader):
			
			signals = torch.squeeze(signals.to(device))

			outputs = model(signals)
			reconstructed = outputs[0]

			recon_loss = criterion(reconstructed, signals)
			loss = model.loss_function(signals, outputs)

			mse_sum += recon_loss.item() * signals.shape[0]
			mse_n += signals.shape[0]

			loader.set_description(
				(
					f'EVAL epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
					f'loss: {loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
				)
			)

			running_recon_loss += recon_loss.item()
			running_loss += loss.item()

			if i % log_interval == 0:
				if i != 0:
					# take the average of running loss
					running_loss /= log_interval
					running_recon_loss /= log_interval


				save_checkpoint_metrics(
					writer=writer,
					model=model,
					sample=signals,
					save_filename=save_filename,
					epoch=epoch,
					iteration=epoch * len(loader) + i,
					loss=running_loss, 
					recon_loss=running_recon_loss,
					prefix='eval',
				)

				running_recon_loss = 0
				running_loss = 0