import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def get_data_loader_X_C_Y(X, C, Y, batch_size):
    '''
    To improve, allow for batch loading of data for large data
    '''
    if C is None:
        C = np.empty((X.shape[0], 0))

    if Y is None:
        Y = np.empty((X.shape[0], 0))
        
    X = torch.tensor(X).float()
    C = torch.tensor(C).float()
    Y = torch.tensor(Y).long()

    data = TensorDataset(X, C, Y)

    loader = DataLoader(data,
                        batch_size=batch_size,
                        num_workers=8)
    return loader


def get_train_val_loaders_X_C_Y(X, Y=None, C=None, batch_size=32, shuffle=False, validation_size=0.1):
    if C is None:
        C = np.empty((X.shape[0], 0))
    if Y is None:
        Y = np.empty((X.shape[0], 0))

    X_train, X_val, C_train, C_val, Y_train, Y_val = train_test_split(
        X, C, Y, test_size=validation_size, shuffle=shuffle)

    train_loader = get_data_loader_X_C_Y(X_train, C_train, Y_train, batch_size)

    val_loader = get_data_loader_X_C_Y(X_val, C_val, Y_val, batch_size=X_val.shape[0])
    
    return train_loader, val_loader