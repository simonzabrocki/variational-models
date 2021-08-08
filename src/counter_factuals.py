from src.sampling import outputs_given_z_c_y
from scipy.spatial import distance
import pandas as pd
import torch
import numpy as np
from torch import nn



def outputs_counterfact_given_y_c(model, y_c_list,  center_z=False):
    '''
    Samples from a list of target and condition given a model

    Parameters:

    Model: VariationalModel
        A fitted VariationalModel 
    y_c_list: list of tensor
        A list of tuple of target and condition tensors
    center_z: bool
        Whether to sample from the origin of the latent space
    
    '''
    n_sample = y_c_list[0][0].shape[0]

    if center_z:
        z = torch.zeros(n_sample, model.latent_shape)

    else:
        z = model.latent_distribution.sample((n_sample, ))
        
    counterfactuals = [outputs_given_z_c_y(model, z, c, y) for y, c in y_c_list]
    
    return counterfactuals  


def P_X_cat_counterfact_given_y_c(model, y_c_list,  center_z=False):
    '''
    Samples counterfactuals from a list of target and condition given a model. 

    Parameters:

    Model: VariationalModel
        A fitted VariationalModel 
    y_c_list: list of tensor
        A list of tuple of target and condition tensors
    center_z: bool
        Whether to sample from the origin of the latent space
    
    '''
    counterfact_outputs = outputs_counterfact_given_y_c(model, y_c_list,  center_z)
    counterfact_probas = torch.cat([nn.Softmax(dim=1)(el['X_logits']) for el in counterfact_outputs]).reshape(len(y_c_list), -1, model.X_categorical_n_classes, model.X_categorical_shape).permute(0, 1, 3, 2)       
    return counterfact_probas



def compute_categorical_average_effect(probas):
    '''
    To be improved (for loop + nan problem + allow for choosing the similarity/distance)
    Pb with jensenshannon distance, some probas that are really close return nan instead of zeros.
    To allow for computation on other kinds of distance for continuous effects
    '''
    if isinstance(probas, torch.Tensor):
        probas = probas.detach().numpy()

    ae = {}
    n_classes = probas.shape[0]

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            jsd = distance.jensenshannon(probas[i].T, probas[j].T, base=3)
            ae[f'ae {i}-{j}'] = np.nanmean(jsd, axis=1)  # gets nan instead of 0 for proba that are really close

    return pd.DataFrame(ae).rename_axis(index='X_categorical_idx').reset_index()