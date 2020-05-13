import numpy as np
import torch

def save_gens_samples(model, save_filename, z_dim, n_samples=16, cuda=torch.cuda.is_available()):
	with torch.no_grad():
		sample = torch.randn(n_samples, z_dim)
		if cuda:
			sample = sample.cuda()
		sample = model.decode(sample).cpu()
		np.save(save_filename,
				   sample.view(16, 784).numpy())

def save_reconstruction(x, x_recon, save_filename):
	x = x.cpu().numpy()
	x_recon = x_recon.cpu().numpy()
	combined = np.vstack((x, x_recon))
	np.save(save_filename, combined)