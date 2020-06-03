
# NO LONGER NEEDED SINCE MAIN RECON 1C CAN RUN WIH CONV MODEL


from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../data/")
from utils import save_gens_samples, save_reconstruction, loss_function_vanilla
from conv_VAE import ConvVAE
from load_EEGs_1c import EEGDataset1c
from constants import *

torch.manual_seed(1)

batch_size = 64
epochs = 1000
num_examples = -1 
cuda = torch.cuda.is_available()
log_interval = 10
z_dim = 32
files_csv = ALL_FILES_SEC

dataset = EEGDataset1c(files_csv, max_num_examples=num_examples, length=784)
train_loader = data.DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=batch_size,
    pin_memory=cuda
  )

eval_loader = train_loader

model = ConvVAE(num_channels=1)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        
        if cuda:
            data = data.cuda()
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function_vanilla(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def eval(epoch):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(eval_loader):

            if cuda:
                data = data.cuda()

            recon_batch, mu, logvar = model(data)
            eval_loss += loss_function_vanilla(recon_batch, data, mu, logvar).item()
            if i == 0:
                save_reconstruction(data.view(-1, 784)[:16], recon_batch[:16], "results_recon/conv_eval_recon_" + str(epoch))

    eval_loss /= len(eval_loader.dataset)
    print('====> eval set loss: {:.4f}'.format(eval_loss))

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        eval(epoch)
        save_filename = 'results_recon/conv_sample_' + str(epoch)
        save_gens_samples(model, save_filename, z_dim, n_samples=16)

