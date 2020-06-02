import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from metrics_utils import get_recon_metrics

def train(epoch, loader, model, optimizer, scheduler, device, writer, log_interval=10, criterion=nn.MSELoss(), save_filename="results_recon/sample"):
	model.train()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	for i, (signals) in enumerate(loader):
		model.zero_grad()

		signals = signals.to(device).view(signals.shape[0], -1)

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

		running_recon_loss += (recon_loss.item() * signals.shape[0]) / signals.shape[0]
		running_loss += (loss.item() * signals.shape[0]) / signals.shape[0]

		if i % log_interval == 0 and i != 0:
			model.eval()

			iteration = epoch * len(loader) + i

			# sample = signals[:sample_size]
			sample = signals

			with torch.no_grad():
				out = model(sample)
				reconstructed = out[0]
				reconstructed, sample = reconstructed.view(reconstructed.shape[0], -1).cpu().numpy(), sample.view(reconstructed.shape[0], -1).cpu().numpy()
				combined = np.vstack((sample[:sample_size], reconstructed[:sample_size]))

				np.save(save_filename + "_train_" + str(epoch), combined)

				fft_diff, hist_diff = get_recon_metrics(reconstructed, sample)
				writer.add_scalar('train/fft diff', fft_diff, iteration)
				writer.add_scalar('train/hist diff', hist_diff, iteration)
				writer.add_scalar('train/recon loss', running_recon_loss / log_interval, iteration)
				writer.add_scalar('train/loss', running_loss / log_interval, iteration)

			running_recon_loss = 0
			running_loss = 0
			model.train()

def eval(epoch, loader, model, device, writer, log_interval=10,  criterion=nn.MSELoss(), save_filename="results_recon/sample"):
	model.eval()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	with torch.no_grad():
		for i, (signals) in enumerate(loader):
			
			signals = signals.to(device).view(signals.shape[0], -1)

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

			running_recon_loss += (recon_loss.item() * signals.shape[0]) / signals.shape[0]
			running_loss += (loss.item() * signals.shape[0]) / signals.shape[0]


			if i % log_interval == 0:
				iteration = epoch * len(loader) + i

				# sample = signals[:sample_size]
				sample = signals

				with torch.no_grad():
					out = model(sample)
					reconstructed = out[0]
					reconstructed, sample = reconstructed.view(sample.shape[0], -1).cpu().numpy(), sample.view(sample.shape[0], -1).cpu().numpy()
					combined = np.vstack((sample[:sample_size], reconstructed[:sample_size]))

					np.save(save_filename + "_eval_" + str(epoch), combined)

					fft_diff, hist_diff = get_recon_metrics(reconstructed, sample)
					writer.add_scalar('eval/fft diff', fft_diff, iteration)
					writer.add_scalar('eval/hist diff', hist_diff, iteration)
					writer.add_scalar('eval/recon loss', running_recon_loss / log_interval, iteration)
					writer.add_scalar('eval/loss', running_loss / log_interval, iteration)

				running_recon_loss = 0
				running_loss = 0