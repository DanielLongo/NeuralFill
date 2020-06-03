# https://github.com/sksq96/pytorch-vae/blob/master/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
	def forward(self, input, size=256):
		return input.view(input.size(0), size, 1, 1)


class TransformToOriginal(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1, 784)


class Transform3D(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1, 28, 28)


class ConvVAE(nn.Module):
	def __init__(self, num_channels=1, num_channels_out=-1, h_dim=256, z_dim=32, use_cuda=torch.cuda.is_available()):
		super(ConvVAE, self).__init__()

		if num_channels_out == -1:
			num_channels_out = num_channels
			
		self.use_cuda = use_cuda

		self.encoder = nn.Sequential(
			Transform3D(),
			nn.Conv2d(num_channels, 32, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=2, stride=2),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=2, stride=1),
			nn.ReLU(),
			Flatten()
		)
		self.fc1 = nn.Linear(h_dim, z_dim)
		self.fc2 = nn.Linear(h_dim, z_dim)
		self.fc3 = nn.Linear(z_dim, h_dim)
		
		self.decoder = nn.Sequential(
			UnFlatten(),
			nn.ConvTranspose2d(h_dim, 128, kernel_size=3, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
			nn.ReLU(),
			nn.ConvTranspose2d(32, num_channels_out, kernel_size=6, stride=2),
			TransformToOriginal(),
			nn.Sigmoid(),
		)
		
	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		# return torch.normal(mu, std)
		esp = torch.randn(*mu.size())
		if self.use_cuda:
			esp = esp.cuda()
		z = mu + std * esp
		return z
	
	def bottleneck(self, h):
		mu, logvar = self.fc1(h), self.fc2(h)
		z = self.reparameterize(mu, logvar)
		return z, mu, logvar
		

	def forward(self, x):
		h = self.encoder(x)
		z, mu, logvar = self.bottleneck(h)
		z = self.fc3(z)
		return self.decoder(z), mu, logvar

	def encode(self, x):
		return self.bottleneck(self.encoder(x))[0]

	def decode(self, z):
		a = self.fc3(z)
		return self.decoder(a)

	def loss_function(self, signals, outputs):
		recon_x, mu, logvar = outputs
		BCE = F.binary_cross_entropy(recon_x.view(recon_x.shape[0], -1), signals.view(signals.shape[0], -1)) # , reduction='sum')
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD


if __name__ == "__main__":
	# x = torch.zeros(64, 1, 28, 28)
	x = torch.zeros(64, 1, 784)
	model = ConvVAE(num_channels=1)
	a = model(x)
	print("a", a[0].shape)

