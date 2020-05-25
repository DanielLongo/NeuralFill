# from https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from nearest_embed import NearestEmbed, NearestEmbedEMA


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist (originally)"""
    def __init__(self, length=784, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.length = length

        self.emb_size = k
        self.fc1 = nn.Linear(length, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, length)

        self.emb = NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, self.length))
        z_q, _ = self.emb(z_e, weight_sg=True)
        z_q = z_q.view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach())
        emb = emb.view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size, int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        # print(torch.sum(abs(x)))
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, self.length), reduction='sum')
        # print("z_e.detach()", z_e.detach().shape)
        self.vq_loss = F.mse_loss(emb, z_e.detach().view(emb.shape[0], -1))
        self.commit_loss = F.mse_loss(z_e.view(emb.shape[0], -1), emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}

if __name__ == "__main__":
    model = VQ_VAE()
    x = torch.zeros((64, 784))
    print(x.shape)
    model(x)
    model.decode(torch.zeros((64, 200)))


