import torch
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
save_index = 1

def train(dataloader, model, loss_fn, optimizer, writer, record_batches=500):
    global save_index
    num_batches = len(dataloader)
    epoch_loss = 0
    record_loss = 0
    batch_count = 0
    
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
        record_loss += math.sqrt(loss.item())
        batch_count += 1
        if batch_count == record_batches:
            writer.add_scalar("loss/batches", record_loss / record_batches, save_index)
            save_index += 1
            record_loss = 0
            batch_count = 0

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