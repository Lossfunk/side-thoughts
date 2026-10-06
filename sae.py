import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.float32

class TopKSAE(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.sae_dim = 8 * input_dim
        self.k = k
        self.encoder = nn.Linear(self.input_dim, self.sae_dim, bias = True)
        self.decoder = nn.Linear(self.sae_dim, self.input_dim, bias = True)
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        values, indices = torch.topk(x, self.k, dim=-1)
        sparse_acts = torch.zeros_like(x)
        sparse_acts.scatter_(dim=-1, index=indices, src=values)
        return sparse_acts
    
    def decode(self, z):
        recon = F.linear(
            z,
            self.decoder.weight,
            self.decoder.bias,
        )
        return recon
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

def sae_loss(x, recon, z, recon_coeff=1.0, l1_coeff=1e-4):
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

def train_sae(activations, input_dim, k=64, batch_size=256, lr=3e-4, epochs=100, l1_coeff=1e-4, device="cuda"):
    print(f"[SAE]: Training started . . .")
    activations = activations.to(precision)
    dataset = TensorDataset(activations)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = TopKSAE(input_dim=input_dim, k=k).to(device).to(precision)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            z, recon = model(x)
            loss, metrics = sae_loss(x, recon, z, l1_coeff=l1_coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (len(loader) + 1e-8)
        print(f"[SAE]: Epoch {epoch+1:02d} | " f"Loss: {avg_loss:.6f} | " f"Recon: {metrics['recon_loss']:.6f} | " f"Sparse: {metrics['sparsity_loss']:.6f}")
    
    print(f"[SAE]: Training Completed.")

    return model


def get_sae_steering_vector(model, pos_activation, neg_activation):
    x = (pos_activation - neg_activation).mean(0).to(next(model.parameters()).device).to(precision)
    z, recon = model.forward(x.unsqueeze(0))
    return recon[0]

