# a larger vesion of VAE1c

import torch
from torch import nn
from torch.nn import functional as F

# basic vae from https://github.com/pytorch/examples/blob/master/vae/main.py

class VAE1cM(nn.Module):
	def __init__(self, z_dim=20):
		super(VAE1c, self).__init__()

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

if __name__ == "__main__":
	model = VAE1c()
	sample_input = torch.zeros((64, 1, 784))
	out = model(sample_input)
	print(out[0].shape)