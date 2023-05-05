import torch
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def train(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    epoch_loss = 0

    model.train()

    for X, y in tqdm(dataloader):
        y = y.unsqueeze(1)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += math.sqrt(loss.item())

    avg_epoch_loss = epoch_loss / num_batches

    return avg_epoch_loss

def val(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            y = y.unsqueeze(1)
            X, y = X.to(device), y.to(device)

            pred = model(X)

            epoch_loss += math.sqrt(loss_fn(pred, y).item())
            
    avg_epoch_loss = epoch_loss / num_batches
      
    return avg_epoch_loss

def test(dataloader, model):
    predictions = []
    
    model.eval()

    with torch.no_grad():
        for X in tqdm(dataloader):
            X = X.to(device)

            pred = model(X)
            predictions.append(pred.cpu())
            
    predictions = torch.cat(predictions, dim=0)  
    return predictions.numpy()