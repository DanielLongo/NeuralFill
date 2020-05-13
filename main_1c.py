# https://github.com/pytorch/examples/blob/master/vae/main.py
import sys
import torch
from torch import optim
import torch.utils.data as data
from torch.nn import functional as F
from torchvision import datasets, transforms

sys.path.append("models/")
sys.path.append("data/")
from VAE1c import VAE1c
from load_EEGs_1c import EEGDataset1c
from constants import *


batch_size = 128
num_epochs = 1000
log_interval = 5
cuda = torch.cuda.is_available()

files_csv = ALL_FILES_SEC
dataset = EEGDataset1c(files_csv, max_num_examples=64*10, length=784)

loader = data.DataLoader(dataset=dataset,
	shuffle=True,
	batch_size=batch_size,
	)

# loader = torch.utils.data.DataLoader(
#	 datasets.MNIST('./mnist', train=True, download=True,
#					transform=transforms.ToTensor()),
#	 batch_size=64, shuffle=True)

model = VAE1c()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if cuda:
	model.cuda()

def loss_function(recon_x, x, mu, logvar):
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	return BCE + KLD

def train(epoch):
	model.train()
	train_loss = 0
	for batch_idx, (data) in enumerate(loader):
		if cuda:
			data = data.cuda()
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(data)
		loss = loss_function(recon_batch, data, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(loader.dataset),
				100. * batch_idx / len(loader),
				loss.item() / len(data)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(loader.dataset)))

for epoch in range(num_epochs):
	train(epoch)


