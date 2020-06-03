import numpy as np
import torch
from torch.nn import functional as F
import os

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_vanilla(recon_x, x, mu, logvar):
	# print("recon_x shape", recon_x.shape)
	# print("x shape", x.shape)
	BCE = F.binary_cross_entropy(recon_x.view(recon_x.shape[0], -1), x.view(x.shape[0], -1))#, reduction='sum')
	# BCE = F.mse_loss(recon_x.view(recon_x.shape[0], -1), x.view(x.shape[0], -1))
	# BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
	
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	return BCE + KLD

def save_gens_samples(model, save_filename, z_dim, n_samples=16, cuda=torch.cuda.is_available()):
	with torch.no_grad():
		sample = torch.randn(n_samples, z_dim)
		if cuda:
			sample = sample.cuda()
		sample = model.decode(sample).cpu()
		np.save(save_filename,
				   sample.view(16, 784).numpy())

def get_gens_samples(model, z_dim, n_samples=16, cuda=torch.cuda.is_available()):
	with torch.no_grad():
		sample = torch.randn(n_samples, z_dim)
		if cuda:
			sample = sample.cuda()
		sample = model.decode(sample).cpu()
		return sample.view(16, 784).numpy()

def save_gens_samples_mc(model, save_filename, z_dim, n_samples=16, cuda=torch.cuda.is_available()):
	with torch.no_grad():
		sample = torch.randn(n_samples, 1, z_dim)
		if cuda:
			sample = sample.cuda()
		sample = model.decode(sample).cpu()
		np.save(save_filename,
				   sample.view(16, -1, 784).numpy())

def save_reconstruction(x, x_recon, save_filename):
	x, x_recon = x.detach().cpu().numpy(), x_recon.detach().cpu().numpy()
	x_recon = np.squeeze(x_recon)
	combined = np.vstack((x, x_recon))
	np.save(save_filename, combined)

def save_reconstruction_mc_to_1cOut(x, x_recon, save_filename, axis=1):
	x, x_recon = x.detach().cpu().numpy(), x_recon.detach().cpu().numpy()
	combined = np.concatenate((x, x_recon), axis=axis)
	np.save(save_filename, combined)

def save_denoise_reconstruction(x, x_noisy, x_recon, save_filename):
	x, x_noisy, x_recon = x.detach().cpu().numpy(), x_noisy.detach().cpu().numpy(), x_recon.detach().cpu().numpy()
	combined = np.vstack((x, x_noisy))
	combined = np.vstack((combined, x_recon))
	np.save(save_filename, combined)

def custom_norm(x):
	abs_mean = (torch.abs(x).mean())
	if abs_mean == 0:
		return x
	x = x/(abs_mean)
	x = (torch.tanh(x) + 1)/2
	return x

def custom_norm_batch(batch):
	out = []
	for x in batch:
		out.append(custom_norm(x))
	out = torch.stack(out)
	return out

def reduce_channel_batch(batch, channel_index, a=0):
	batch = batch.clone()
	batch[:, channel_index, :] = batch[:, channel_index, :] * a
	return batch

def interpolate_signals(a, b):
	z = (a + b)/2
	return z

def find_valid_filename(target_filename, file_dir):
	if os.path.exists(file_dir + target_filename):
		if str.isnumeric(target_filename[-1]) and target_filename[-3:-1] == '_r':
			return find_valid_filename(target_filename[:-1] + str(int(target_filename[-1]) + 1), file_dir)
		return find_valid_filename(target_filename + '_r1', file_dir)
	return target_filename




