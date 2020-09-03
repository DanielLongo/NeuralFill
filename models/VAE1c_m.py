# a larger vesion of VAE1c

import torch
from torch import nn
from torch.nn import functional as F

# basic vae from https://github.com/pytorch/examples/blob/master/vae/main.py
import torch
from torch import nn
from torch.nn import functional as F

# basic vae from https://github.com/pytorch/examples/blob/master/vae/main.py

class VAE1cM(nn.Module):
	def __init__(self, z_dim=20):
		super(VAE1cM, self).__init__()

		self.encode_nn = nn.Sequential(
			nn.Linear(784, 500),
			nn.ReLU(),
			nn.Linear(500, 300),
			nn.ReLU(),
			nn.Linear(300, 200),
			nn.ReLU(),
		)
		
		self.fc71 = nn.Linear(200, z_dim)
		self.fc72 = nn.Linear(200, z_dim)

		self.decode_nn = nn.Sequential(
			nn.Linear(z_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 300),
			nn.ReLU(),
			nn.Linear(300, 500),
			nn.ReLU(),
			nn.Linear(500, 784),
			nn.Sigmoid()
		)

		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, z_dim)
		self.fc22 = nn.Linear(400, z_dim)
		self.fc3 = nn.Linear(z_dim, 400)
		self.fc4 = nn.Linear(400, 784)
		self.z_dim = z_dim

	def encode(self, x):
		# x = torch.squeeze(x)
		h = self.encode_nn(x)
		return self.fc71(h), self.fc72(h)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		a = self.decode_nn(z)
		return a

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

	def loss_function(self, signals, outputs):
		recon_x, mu, logvar = outputs
		BCE = F.binary_cross_entropy(recon_x.view(recon_x.shape[0], -1), signals.view(signals.shape[0], -1) , reduction='sum')
		
		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD

	def sample(self, num_samples):
		z = torch.randn(num_samples, self.z_dim)
		if torch.cuda.is_available():
			z = z.cuda()
		out = self.decode(z) 
		return out.view(num_samples, -1).detach().cpu().numpy()

if __name__ == "__main__":
	model = VAE1c()
	sample_input = torch.zeros((64, 1, 784))
	out = model(sample_input)
	print(out[0].shape)