import torch
import numpy as np


def outputs_given_z_c_y(model, z, c, y):
    model.eval()

    decoder_output = model.decode(torch.cat([z, y], axis=1), c)

    return decoder_output


def sample_outputs_given_y_and_c(model, y, c, center_z=False):
    n_sample = y.shape[0]

    if center_z:
        z = torch.zeros(n_sample, model.latent_shape)

    else:
        z = model.latent_distribution.sample((n_sample, ))

    return outputs_given_z_c_y(model, z, c, y)


def sample_X_given_y_and_c(model, y, c, center_z=False, argmax_categorical=False, mean_continuous=False):

    outputs = sample_outputs_given_y_and_c(model, y, c, center_z)

    X_sample_list = []
    if model.X_continuous_shape != 0:

        if mean_continuous:
            X_continuous = outputs['X_mu']
        else:
            X_continuous = model.reparameterize(
                outputs['X_mu'], outputs['X_logvar'], 1, model.X_distribution)

        X_sample_list.append(X_continuous.detach().numpy())

    if model.X_categorical_shape != 0:
        # V1
        if argmax_categorical:
            X_categorical = outputs['X_logits'].argmax(axis=1)

        # V2
        else:
            logits = outputs['X_logits'].permute(0, 2, 1).reshape(-1, model.X_categorical_n_classes)
            X_categorical = torch.distributions.categorical.Categorical(logits=logits)\
                                               .sample()\
                                               .reshape(-1, model.X_categorical_shape)

        X_sample_list.append(X_categorical.detach().numpy())

    return np.concatenate(X_sample_list, axis=1)

# import torch


# def P_of_X_given_z_c_y(model, z, c, y):

#     model.eval()

#     decoder_output = model.decode(torch.cat([z, y], axis=1), c)

#     return decoder_output['probas']  # .argmax(axis=1)


# def sample_P_of_X_given_y_and_c(model, y, c, center_z=False):
#     n_sample = y.shape[0]

#     if center_z:
#         z = torch.zeros(n_sample, model.latent_shape)

#     else:
#         z = model.latent_distribution.sample((n_sample, ))

#     return P_of_X_given_z_c_y(model, z, c, y)


# def sample_P_of_X_for_all_y_given_c(model, c):
#     '''
#     TO CLEAN UP
#     Sample X from P(X|Y,C) for all possible Y.
#     '''
#     X_shape = model.hparams['X_shape']
#     n_classes = model.hparams['n_classes']
#     n_classes_decoder = model.hparams['n_classes_decoder']
#     n_sample_per_class = c.shape[0]

#     # to improve
#     y = torch.zeros((n_classes * n_sample_per_class, n_classes))
#     for classe in range(n_classes):
#         begin = classe * n_sample_per_class
#         end = n_sample_per_class * (classe + 1)
#         y[begin:end, classe] = 1

#     X = sample_P_of_X_given_y_and_c(model, y, c.repeat(n_classes, 1))
#     X = X.reshape(n_classes, -1, n_classes_decoder, X_shape)

#     return X.permute(0, 1, 3, 2)


# def sample_X_for_all_y_given_c(model, c=None):
#     P_of_X_sample = sample_P_of_X_for_all_y_given_c(model, c)
#     X_sample = torch.distributions.categorical.Categorical(P_of_X_sample.reshape(-1, model.hparams.n_classes_decoder)).sample().reshape(-1, model.hparams.X_shape).numpy()
#     return X_sample


# # BELOW: to be updated using the above function


# def X_given_z_c_y(model, z, c, y):

#     p_x = P_of_X_given_z_c_y(model, z, c, y)

#     return p_x.argmax(axis=1)  # For now we take the max but latter should be sampled as well here !


# def sample_X_given_y_and_c(model, y, c, center_z=False):

#     n_sample = y.shape[0]

#     if center_z:
#         z = torch.zeros(n_sample, model.latent_shape)

#     else:
#         z = model.latent_distribution.sample((n_sample, ))

#     return X_given_z_c_y(model, z, c, y)


# def sample_X_for_all_y_given_c(model, c):
#     '''
#     Sample X from P(X|Y,C) for all possible Y.
#     '''
#     X_shape = model.hparams['X_shape']
#     n_classes = model.hparams['n_classes']
#     n_sample_per_class = c.shape[0]
#
#     # to improve
#     y = torch.zeros((n_classes * n_sample_per_class, n_classes))
#     for classe in range(n_classes):
#         begin = classe * n_sample_per_class
#         end = n_sample_per_class * (classe + 1)
#         y[begin:end, classe] = 1
#
#     X = sample_X_given_y_and_c(model, y, c.repeat(n_classes, 1))
#
#     return X.reshape(n_classes, -1, X_shape)
# 