from torch import nn
from collections import OrderedDict


def dense_norm_layer(i, in_features, out_features, activation):
    linear_layer = nn.Linear(in_features, out_features)
    batch_norm_layer = nn.BatchNorm1d(num_features=out_features)

    layer = OrderedDict({f'dense_layer_{i}': linear_layer,
                         f'batch_norm_{i}': batch_norm_layer,
                         f'activation_{i}': activation})
    return layer


def dense_layer(i, in_features, out_features, activation):
    linear_layer = nn.Linear(in_features, out_features)
    layer = OrderedDict({f'dense_layer_{i}': linear_layer,
                         f'activation_{i}': activation})
    return layer


def mean_var_layer(in_features, out_features):
    mean_logvar_layer = nn.Linear(in_features, 2 * out_features)
    layer = OrderedDict({
        'mean_logvar_layer': mean_logvar_layer})
    return layer


def meanlogvar_class_layer(in_features, out_features):
    mean_logvar_layer = nn.Linear(in_features, out_features)
    layer = OrderedDict({'meanlogvar_class_layer': mean_logvar_layer})
    return layer


def multinomial_proba_layer(in_features, out_features):
    multinomial_proba_layer = nn.Linear(in_features, out_features)
    layer = OrderedDict({'multinomial_proba_layer': multinomial_proba_layer})
    return layer
