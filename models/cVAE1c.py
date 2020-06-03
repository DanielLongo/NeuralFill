import torch
from torch import nn
from torch.nn import functional as F

# basic vae from https://github.com/pytorch/examples/blob/master/vae/main.py

class cVAE1c(nn.Module):
	def __init__(self, z_dim=20):
		super(cVAE1c, self).__init__()

		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, z_dim)
		self.fc22 = nn.Linear(400, z_dim)
		self.fc3 = nn.Linear(z_dim * 3, 400)
		self.fc4 = nn.Linear(400, 784)

	def encode(self, x):
		# x = torch.squeeze(x)
		h1 = F.relu(self.fc1(x))
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def forward(self, recording):
		# x (bsize, 3, length)
		# x[1] is the bad channel
		# x[0] and x[2] are the good channels
		x = recording[:, 1, :]
		y_1 = recording[:, 0, :]
		y_2 = recording[:, 2, :]

		mu, logvar = self.encode(x.view(-1, 784))
		z = self.reparameterize(mu, logvar)
		
		mu_y_0, logvar_y_0 = self.encode(y_1)
		mu_y_1, logvar_y_1 = self.encode(y_2)
		
		
		# y_0 = mu_y_0
		# y_1 = mu_y_1
		y_0 = self.reparameterize(mu_y_0, logvar_y_0)
		y_1 = self.reparameterize(mu_y_1, logvar_y_1)
		
		y = torch.cat([y_0, y_1], 1)
		z = torch.cat([z, y], 1)
		
		return self.decode(z), mu, logvar

	def loss_function(self, signals, outputs):
		recon_x, mu, logvar = outputs
		BCE = F.binary_cross_entropy(recon_x.view(recon_x.shape[0], -1), signals.view(signals.shape[0], -1)) # , reduction='sum')
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return BCE + KLD

if __name__ == "__main__":
	model = cVAE1c()
	# sample_y = torch.zeros((64, 2, 784))
	sample_input = torch.zeros((64, 3, 784))
	out = model(sample_input)#, sample_y)
	print(out[0].shape)