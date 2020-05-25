# https://github.com/pytorch/examples/blob/master/vae/main.py
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../models/")
sys.path.append("../data/")
sys.path.append("../metrics/")
from metrics_utils import get_metrics
from utils import save_gens_samples, save_reconstruction, loss_function_vanilla, find_valid_filename
from VQ_VAE_1c import VQ_VAE
from load_EEGs_1c import EEGDataset1c
from constants import *

torch.manual_seed(1)
target_filename = "nn-vq_s_1c"
runs_dir = "runs"
run_filename = find_valid_filename(target_filename, HOME_PATH + 'reconstruction/' + runs_dir + '/')
print("Run Filname:", run_filename)


params = {
    "batch_size": 128,
    "epochs": 1000,
    "num_examples_train": -1, #128*40,
    "num_examples_eval": -1, #128*10,
    "cuda": torch.cuda.is_available(),
    "log_interval": 1e100,
    "z_dim": 60, # must be multiple of 10 b/c k=10
    "length": 784,
    "tensorboard_log_interval": 25,
    "run_filename": run_filename,
    "lr" : 1e-4
}

model_save_filename = "saved_models/" + run_filename
writer = SummaryWriter(runs_dir + '/' + run_filename)

train_dataset = EEGDataset1c(TRAIN_FILES_CSV, max_num_examples=params["num_examples_train"], length=params["length"])
train_loader = data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=params["batch_size"],
    pin_memory=params["cuda"]
  )

eval_dataset = EEGDataset1c(DEV_FILES_CSV, max_num_examples=params["num_examples_eval"], length=params["length"])
eval_loader = data.DataLoader(
    dataset=eval_dataset,
    shuffle=True,
    batch_size=params["batch_size"],
    pin_memory=params["cuda"]
  )

model = VQ_VAE(hidden=params["z_dim"])

if params["cuda"]:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=params["lr"])


def train(epoch):
    model.train()
    train_loss = 0
    running_loss = 0.0 # for tb
    for batch_idx, (data) in enumerate(train_loader):
        
        if params["cuda"]:
            data = data.cuda()
        
        optimizer.zero_grad()

        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()

        latest_losses = model.latest_losses()
        train_loss += latest_losses["cross_entropy"].item()
        running_loss += latest_losses["cross_entropy"].item()
        optimizer.step()

        if batch_idx % params["tensorboard_log_interval"] == params["tensorboard_log_interval"] - 2:
            hist_diff, spec_diff = get_metrics(model, dataset=train_dataset, z_dim=params["z_dim"], n_dataset=100, n_model=100, print_results=False)

            iteration = epoch * len(train_loader) + batch_idx
            
            writer.add_scalar('train/hist diff', hist_diff, iteration)
            writer.add_scalar('train/spec diff', spec_diff, iteration)
            writer.add_scalar('train/loss', running_loss / params["tensorboard_log_interval"], iteration)
                
            running_loss = 0.0

        # if batch_idx % params["log_interval"] == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss


def eval(epoch):
    model.eval()
    running_loss = 0.0
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(eval_loader):

            if params["cuda"]:
                data = data.cuda()

            outputs = model(data)
            loss = model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            eval_loss += latest_losses["cross_entropy"].item()
            running_loss += latest_losses["cross_entropy"].item()
            
            if batch_idx % params["tensorboard_log_interval"] == params["tensorboard_log_interval"] - 1:
                hist_diff, spec_diff = get_metrics(model, dataset=eval_dataset, z_dim=params["z_dim"], n_dataset=100, n_model=100, print_results=False)


                iteration = epoch * len(eval_loader) + batch_idx
                
                writer.add_scalar('eval/hist diff', hist_diff, iteration)
                writer.add_scalar('eval/spec diff', spec_diff, iteration)
                writer.add_scalar('eval/loss', running_loss / params["tensorboard_log_interval"], iteration)


                running_loss = 0.0

            if batch_idx == 0:
                recon_batch = outputs[0]
                save_reconstruction(data.view(-1, 784)[:16], recon_batch[:16], "results_recon/eval_recon_" + str(epoch))

    eval_loss /= len(eval_loader.dataset)
    print('====> eval set loss: {:.4f}'.format(eval_loss))
    return eval_loss

if __name__ == "__main__":
    sample_batch = iter(train_loader).next()
    if params["cuda"]:
        sample_batch = sample_batch.cuda()
    writer.add_graph(model, sample_batch)
    writer.close()
    cur_lowest_eval_loss = 1e100
    for epoch in range(1, params["epochs"] + 1):

        try:
            train_loss = train(epoch)
            eval_loss = eval(epoch)

            if eval_loss < cur_lowest_eval_loss:
                cur_lowest_eval_loss = eval_loss

        except KeyboardInterrupt:
            print("Ending Early")
            params["epochs"] = epoch
            break

    if eval_loss < cur_lowest_eval_loss:
        cur_lowest_eval_loss = eval_loss


    writer.add_hparams({
            "epochs": params["epochs"],
            "lr" : params["lr"],
            "z_dim": params["z_dim"],
            "m_train": len(train_dataset),
            "bsize": params["batch_size"],
            "length": params["length"]
        },
        {
            'hparam/loss': cur_lowest_eval_loss
        }
    )
    writer.close()
    torch.save(model, model_save_filename)
    print("FINISHED", run_filename)
            # get_metrics(model=model_save_filename, dataset=dataset, z_dim=z_dim, print_results=True)

