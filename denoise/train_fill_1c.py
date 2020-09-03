import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import random
from inspect import signature
import sys
sys.path.append("../")
sys.path.append("../metrics/")
sys.path.append("../artifacts")

from synthetic_artifacts_1c import SyntheticArtifiactsLabeled1c
from metrics_utils import save_checkpoint_metrics
from utils import reduce_channel_batch
from data import normalize

target_artifacts = {
		"no signal": False,
		"60hz noise": True,
		"blink": True
	}
blinks = SyntheticArtifiactsLabeled1c(20, length=784, target_artifacts=target_artifacts, normalize=normalize, label=False)

def distort_channel_batch(signals, distorted_channel_index):
	for i in range(signals.shape[0]):
		cur_blinks = (blinks[random.randint(0,19)][0]).cuda()
		signals[i][distorted_channel_index] = (signals[i][distorted_channel_index] + cur_blinks) / 2
	return signals

def model_is_conditional(model):
	# checks if model recieves conditional labels marking bad channels
	sig = signature(model.forward)
	params = sig.parameters 
	if "noisy_labels" in list(params):
		return True
	return False

def train(epoch, loader, model, optimizer, scheduler, device, writer, log_interval=10, criterion=nn.MSELoss(), save_filename="results_fill/sample"):
	model.train()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0
	num_examples = 0

	add_channels_distorted = model_is_conditional(model)

	for i, (signals) in enumerate(loader):
		model.zero_grad()

		distorted_channel_index = random.randint(0, signals.shape[1] - 1) # 1
	
		signals = signals.to(device).view(signals.shape[0], signals.shape[1], -1)
		target_channel = signals[:, distorted_channel_index, :]

		# signals_reduced = reduce_channel_batch(signals, distorted_channel_index, a=0) # reduce distorted channel to 0
		signals_reduced = distort_channel_batch(signals, distorted_channel_index)
		
		if add_channels_distorted:
			channels_distorted = [0 for i in range(signals.shape[1])]
			channels_distorted[distorted_channel_index] = 1 # 1 for bad channel 0 for good
			outputs = model(signals_reduced, noisy_labels=channels_distorted)
		else:
			channels_distorted = None
			outputs = model(signals_reduced)

		reconstructed = outputs[0] # first output is always the recon 
		if reconstructed.shape[1] > 1:
			# multiple channels generated but only want to compare noisy channel 
			# reconstructed = torch.squeeze(reconstructed[:, 1, :])
			# outputs = (reconstructed,) + outputs[1:] 

			# multiple channels generated and want to compare all channels
			target_channel = signals

		reconstructed = torch.squeeze(reconstructed)
		target_channel = torch.squeeze(target_channel)

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

		num_examples += signals.shape[0]

		if i % log_interval == 0 and i != 0:
			model.eval()

			num_channels = reconstructed.shape[1]

			save_checkpoint_metrics(
				writer=writer,
				model=model,
				sample=signals,
				save_filename=save_filename,
				epoch=epoch,
				iteration=epoch * len(loader) + i,
				loss=running_loss / num_examples, 
				recon_loss=running_recon_loss / num_examples,
				prefix='train',
				noisy_labels=channels_distorted
			)

			running_recon_loss = 0
			running_loss = 0
			num_examples = 0
			model.train()

def eval(epoch, loader, model, device, writer, log_interval=10,  criterion=nn.L1Loss(), save_filename="results_fill/sample"):
	model.eval()
	loader = tqdm(loader)

	sample_size = 16

	mse_sum = 0
	mse_n = 0

	running_loss = 0
	running_recon_loss = 0

	add_channels_distorted = model_is_conditional(model)

	num_examples = 0

	with torch.no_grad():
		for i, (signals) in enumerate(loader):
			distorted_channel_index = random.randint(0, signals.shape[1] - 1)
			
			signals = signals.to(device).view(signals.shape[0], signals.shape[1], -1)
			target_channel = signals[:, distorted_channel_index, :]
			# signals_reduced = reduce_channel_batch(signals, distorted_channel_index, a=0) # reduce distorted channel to 0
			signals_reduced = distort_channel_batch(signals, distorted_channel_index)

			if add_channels_distorted:
				channels_distorted = [0 for i in range(signals.shape[1])]
				channels_distorted[distorted_channel_index] = 1 # 1 for bad channel 0 for good
				outputs = model(signals_reduced, noisy_labels=channels_distorted)
			else:
				channels_distorted = None
				outputs = model(signals_reduced)

			reconstructed = outputs[0]
			if reconstructed.shape[1] > 1:
				# reconstructed = torch.squeeze(reconstructed[:, 1, :])
				# outputs = (reconstructed,) + outputs[1:] 

				# multiple channels generated and want to compare all channels
				target_channel = signals


			reconstructed = torch.squeeze(reconstructed)
			target_channel = torch.squeeze(target_channel)
				
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

			running_recon_loss += recon_loss.item() #(recon_loss.item() * signals.shape[0]) / signals.shape[0]
			running_loss += loss.item() #(loss.item() * signals.shape[0]) / signals.shape[0]
			num_examples += signals.shape[0]



			if i % log_interval == 0:

				num_channels = reconstructed.shape[1]

				save_checkpoint_metrics(
					writer=writer,
					model=model,
					sample=signals,
					save_filename=save_filename,
					epoch=epoch,
					iteration=epoch * len(loader) + i,
					loss=running_loss / num_examples, 
					recon_loss=running_recon_loss / num_examples,
					prefix='eval',
					noisy_labels=channels_distorted,
				)

				running_recon_loss = 0
				running_loss = 0
				num_examples = 0