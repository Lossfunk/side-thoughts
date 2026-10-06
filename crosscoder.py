import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Gumbel, Cauchy, Normal

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.float32
euler_mascheroni = 0.5772156649

class TopKCrossCoder(nn.Module):
    def __init__(self, input_dim, dims, k):
        super().__init__()
        self.input_dim = input_dim
        self.sae_dim = 8 * input_dim
        self.dims = dims
        self.k = k
        self.encoder = nn.Linear(self.input_dim, self.sae_dim, bias = True)
        self.decoder = nn.Linear(self.sae_dim, self.input_dim, bias = True)
        self.small_encoder = nn.Linear(self.dims[0], self.input_dim)
        self.medium_encoder = nn.Linear(self.dims[1], self.input_dim)
        self.large_encoder = nn.Linear(self.dims[2], self.input_dim)
        self.small_decoder = nn.Linear(self.input_dim, self.dims[0])
        self.medium_decoder = nn.Linear(self.input_dim, self.dims[1])
        self.large_decoder = nn.Linear(self.input_dim, self.dims[2])
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        values, indices = torch.topk(x, self.k, dim=-1)
        sparse_acts = torch.zeros_like(x)
        sparse_acts.scatter_(dim=-1, index=indices, src=values)
        beta = sparse_acts.std()
        mu = sparse_acts.mean()
        beta = 0.001 * sparse_acts.std() * (6 ** 0.5) / torch.pi
        mu = sparse_acts.mean() - euler_mascheroni * beta
        gumbel_dist = Gumbel(loc=mu, scale=beta)
        # gumbel_dist = Normal(loc=mu, scale=beta)
        sparse_acts = gumbel_dist.sample((sparse_acts.shape[-1],)).unsqueeze(0)
        return sparse_acts
    
    def decode(self, z):
        recon = F.linear(
            z,
            self.decoder.weight,
            self.decoder.bias,
        )
        return recon
    
    def forward(self, x, upsample_medium = False, upsample_large = False):
        dim = x.shape[-1]
        if dim == self.dims[0]:
            x = self.small_encoder(x)
        elif dim == self.dims[1]:
            x = self.medium_encoder(x)
        else:
            x = self.large_encoder(x)
        z = self.encode(x)
        recon = self.decode(z)
        if upsample_large:
            recon = self.large_decoder(recon)
            return x, recon
        if upsample_medium:
            recon = self.medium_decoder(recon)
            return x, recon
        if dim == self.dims[0]:
            recon = self.small_decoder(recon)
        elif dim == self.dims[1]:
            recon = self.medium_decoder(recon)
        elif dim == self.dims[2]:
            recon = self.large_decoder(recon)
        return z, recon

def crosscoder_loss(x, recon, z, recon_coeff=1.0, l1_coeff=1e-4):
    recon_loss = F.mse_loss(recon, x)
    sparsity_loss = z.abs().mean()
    total_loss = (
        recon_coeff * recon_loss
        + l1_coeff * sparsity_loss
    )
    return total_loss, {
        "recon_loss": recon_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
    }

def train_crosscoder(activations, dims, input_dim, crosscoder_dir, k=64, batch_size=512, lr=1e-4, epochs=50, l1_coeff=1e-4, device="cuda"):
    print(f"[CrossCoder]: Training started . . .")
    small, medium, large = activations
    small = small.to(precision)
    medium = medium.to(precision)
    large = large.to(precision)
    small_dataset = TensorDataset(small)
    medium_dataset = TensorDataset(medium)
    large_dataset = TensorDataset(large)
    small_loader = DataLoader(
        small_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    medium_loader = DataLoader(
        medium_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    large_loader = DataLoader(
        large_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = TopKCrossCoder(input_dim=input_dim, dims=dims, k=k).to(device).to(precision)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    state_dict_path = os.path.join(crosscoder_dir, "crosscoder.pt")
    if os.path.isfile(state_dict_path):
        checkpoint = torch.load(state_dict_path, map_location='cuda:0', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in small_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            z, recon = model(x)
            loss, metrics = crosscoder_loss(x, recon, z, l1_coeff=l1_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(small_loader) + 1e-8)
        print(f"[CrossCoder]: Small | Epoch {epoch+1:02d} | " f"Loss: {avg_loss:.6f} | " f"Recon: {metrics['recon_loss']:.6f} | " f"Sparse: {metrics['sparsity_loss']:.6f}")

        running_loss = 0.0
        for batch in medium_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            z, recon = model(x)
            loss, metrics = crosscoder_loss(x, recon, z, l1_coeff=l1_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(medium_loader) + 1e-8)
        print(f"[CrossCoder]: Medium | Epoch {epoch+1:02d} | " f"Loss: {avg_loss:.6f} | " f"Recon: {metrics['recon_loss']:.6f} | " f"Sparse: {metrics['sparsity_loss']:.6f}")

        running_loss = 0.0
        for batch in large_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            z, recon = model(x)
            loss, metrics = crosscoder_loss(x, recon, z, l1_coeff=l1_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(large_loader) + 1e-8)
        print(f"[CrossCoder]: Large | Epoch {epoch+1:02d} | " f"Loss: {avg_loss:.6f} | " f"Recon: {metrics['recon_loss']:.6f} | " f"Sparse: {metrics['sparsity_loss']:.6f}")

    print(f"[CrossCoder]: Training Completed.")
    torch.save(model.state_dict(), state_dict_path)

    return model


def check_pretrained_crosscoder(dims, input_dim, crosscoder_dir, k = 64, lr=1e-4, device="cuda"):
    model = TopKCrossCoder(input_dim=input_dim, dims=dims, k=k).to(device).to(precision)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    state_dict_path = os.path.join(crosscoder_dir, "crosscoder.pt")
    if os.path.isfile(state_dict_path):
        checkpoint = torch.load(state_dict_path, map_location='cuda:0', weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        return model


def get_crosscoder_steering_vector(model, pos_activation, neg_activation, upsample_medium = False, upsample_large = False):
    x = (pos_activation - neg_activation).to(next(model.parameters()).device).to(precision)
    z, recon = model.forward(x.unsqueeze(0), upsample_medium = upsample_medium, upsample_large = upsample_large)
    return recon[0]

def get_crosscoder_teacher_vector(model, embedding, upsample_medium = False, upsample_large = False):
    x = embedding.to(next(model.parameters()).device).to(precision)
    z, recon = model.forward(x.unsqueeze(0), upsample_medium = upsample_medium, upsample_large = upsample_large)
    return recon[0]
