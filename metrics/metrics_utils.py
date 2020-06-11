import scipy
import scipy.signal
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import torch

def get_samples_real_stack(dataset, n=-1):
	# reutns all examples of dataset in a singe batch
	if n == -1 or n > len(dataset):
		n = len(dataset)
	out = np.zeros(dataset[0].shape)
		
	for i in range(n):
		out = np.vstack((out, dataset[i].numpy()))
	out = out[1:, :]
	return out
	

def get_signal_diff_histogram(signals_a, signals_b, n_bins=50):
	signals_a = signals_a.reshape(-1)
	signals_b = signals_b.reshape(-1)
	hist_a, _ = np.histogram(signals_a, bins=n_bins)
	hist_b, _ = np.histogram(signals_b, bins=n_bins)
	
	hist_a = hist_a / np.linalg.norm(hist_a)
	hist_b = hist_b / np.linalg.norm(hist_b)
	
	return np.sum(np.abs(hist_a - hist_b))
	
def get_signal_diff_spectrogram(signals_a, signals_b):
	m_singals_a = signals_a.shape[0]
	m_singals_b = signals_b.shape[0]
	assert(signals_a.shape[1] == signals_b.shape[1]), "signals should be equal length"
	combined = np.vstack((signals_a, signals_b))
	_, _, combined_spectrogram = compute_spectrogram(combined)
	
	combined_spectrogram = np.sum(combined_spectrogram, axis=2)
	
	spectrogram_a = combined_spectrogram[:m_singals_a]
	spectrogram_b = combined_spectrogram[m_singals_a:]
	
	# Reduce dimension again b/c time dim isn't relevant - just want total amount of freq
	spectrogram_a = np.sum(spectrogram_a, axis=0)
	spectrogram_b = np.sum(spectrogram_b, axis=0)
	
	return np.sum(np.abs(spectrogram_a - spectrogram_b))

def compute_spectrogram(signals, fs=200, graph=False):
	window = ('tukey', .25) # turkey is the default
#	 window = scipy.signal.triang(60)
	f, t, Sxx = scipy.signal.spectrogram(signals, fs, window=window)
	Sxx = Sxx / np.linalg.norm(Sxx)
	if graph:
		plt.pcolormesh(t, f, Sxx)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.show()
	
	return  f, t, np.squeeze(Sxx)

def get_metrics(model, dataset, z_dim, n_dataset=100, n_model=100, print_results=False):
	# model is either the model filename or model itself 

	# works for mc or single channel
	
	# compute samples with loaded model
	if type(model) == str:
		model = torch.load(model)

	sample_generated = torch.randn(n_model, 1, z_dim).cuda()
	sample_generated = model.decode(sample_generated).cpu().detach().numpy()
	sample_generated = sample_generated.reshape(-1, sample_generated.shape[-1])
	
	# grab n exmples from dataset 
	dataset = [x.numpy() for x in dataset]
	random.shuffle(dataset)
	sample_real = np.asarray(dataset[:n_dataset])
	sample_real = sample_real.reshape(-1, sample_real.shape[-1])
	
	assert(sample_real.shape[-1] == sample_generated.shape[-1]), "Signal length should be the same"
	
	
	hist_diff = get_signal_diff_histogram(sample_real, sample_generated)
	spec_diff = get_signal_diff_spectrogram(sample_real, sample_generated)

	if print_results:
		print("HIST DIFF", hist_diff)
		print("SPEC DIFF", spec_diff)

	return hist_diff, spec_diff

def get_recon_diff_fft(x, recon_x):
	fft_x = np.fft.fft(x)
	fft_recon_x = np.fft.fft(recon_x)

	fft_x /= np.sum(fft_x)
	fft_recon_x /= np.sum(fft_recon_x)
	
	return (np.sum(abs(fft_x - fft_recon_x)))


def get_recon_diff_hist(x, recon_x, n_bins=50):
	signals_a = x.reshape(-1)
	signals_b = recon_x.reshape(-1)
	hist_a, _ = np.histogram(signals_a, bins=n_bins)
	hist_b, _ = np.histogram(signals_b, bins=n_bins)
	
	hist_a = hist_a / np.linalg.norm(hist_a)
	hist_b = hist_b / np.linalg.norm(hist_b)
	
	return np.sum(np.abs(hist_a - hist_b))

def get_recon_metrics(x, recon_x):
	fft_diff = get_recon_diff_fft(x, recon_x)
	hist_diff = get_recon_diff_hist(x, recon_x)
	return fft_diff, hist_diff

def save_checkpoint_metrics(writer, model, sample, save_filename, epoch, iteration, loss, recon_loss, prefix, sample_size=16, noisy_labels=None):
	with torch.no_grad():
		if noisy_labels is None:
			out = model(sample)
		else:
			out = model(sample, noisy_labels=noisy_labels)
		reconstructed = out[0]
		reconstructed, sample = torch.squeeze(reconstructed).cpu().numpy(), torch.squeeze(sample).cpu().numpy()
		# reconstructed, sample = reconstructed.view(reconstructed.shape[0], -1).cpu().numpy(), sample.view(reconstructed.shape[0], -1).cpu().numpy()

		if len(sample.shape) == 3 and len(reconstructed.shape) == 3:
			# both sample and reconstructed have multiple channels
			combined = np.concatenate((sample[:sample_size], reconstructed[:sample_size]))

		elif len(sample.shape) == 3:
			# the sample has multiple channels so must expand dim of reconstructed for vstack
			reconstructed = reconstructed.reshape((reconstructed.shape[0], 1, -1))
			combined = np.concatenate((sample[:sample_size], reconstructed[:sample_size]), axis=1)

		else:
			combined = np.vstack((sample[:sample_size], reconstructed[:sample_size]))

		np.save(save_filename + "-" + prefix + "-" + str(epoch), combined)

		fft_diff, hist_diff = get_recon_metrics(reconstructed, sample)

		writer.add_scalar(prefix + '/fft diff', fft_diff, iteration)
		writer.add_scalar(prefix + '/hist diff', hist_diff, iteration)
		writer.add_scalar(prefix + '/recon loss', recon_loss, iteration)
		writer.add_scalar(prefix + '/loss', loss, iteration)

def get_fft(x):
	fft_x = np.fft.fft(x)
	return fft_x/np.sum(fft_x)

def get_hist(x, n_bins=100, range=(-2000,2000)):
	signals_a = x.reshape(-1)
	hist_a, _ = np.histogram(signals_a, bins=n_bins, range=range)
	# hist_a = hist_a / np.linalg.norm(hist_a)
	return hist_a



