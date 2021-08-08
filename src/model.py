from argparse import ArgumentParser
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from src.loss_functions import normal_kld, gaussian_nll, l1_reg, l2_reg
from src.diagnostic_plots import make_diagnostic_plots
from src.layers import dense_norm_layer


def split_X_y_cat_con(model, x, y):
    '''
    Utilities to split continuous and categorical dimensions in the model
    '''
    x_continuous, x_categorical = torch.split(
        x, [model.X_continuous_shape, model.X_categorical_shape], dim=1)
    y_continuous, y_categorical = torch.split(
        y, [model.y_continuous_shape, model.y_categorical_shape], dim=1)
    
    return x_continuous, x_categorical, y_continuous, y_categorical


def format_categorical_output(var_categorical, var_logits, var_name):
    '''
    Format categorical dimensions for output
    '''
    if var_categorical.shape[-1] != 0:
        return {f'{var_name}_categorical_pred': nn.Softmax(dim=1)(var_logits).argmax(dim=1), f'{var_name}_categorical': var_categorical}
    else:
        return {f'{var_name}_categorical_pred': None, f'{var_name}_categorical': None}

    
def format_continuous_output(var_continuous, var_mu, var_name):
    '''
    Format continuous dimensions for output
    '''
    if var_continuous.shape[-1] != 0:
        return {f'{var_name}_continuous_pred': var_mu, f'{var_name}_continuous': var_continuous}
    else:
        return {f'{var_name}_continuous_pred': None, f'{var_name}_continuous': None}
    
    
def forward_format_output(model, X, C, y):
    '''
    Utilities for easy access to usable outputs 
    '''
    model.eval()
    model_outputs = model.forward(X, C)
    
    x_continuous, x_categorical, y_continuous, y_categorical = split_X_y_cat_con(model, X, y)
    
    outputs = {}
    outputs.update({'mu_latent': model_outputs['z']})

    outputs.update(format_continuous_output(
        x_continuous, model_outputs['decoder']['X_mu'], 'x'))
    outputs.update(format_continuous_output(
        y_continuous, model_outputs['encoder']['y_mu'], 'y'))

    outputs.update(format_categorical_output(
        x_categorical, model_outputs['decoder']['X_logits'], 'x'))
    outputs.update(format_categorical_output(
        y_categorical, model_outputs['encoder']['y_logits'], 'y'))
    
    return outputs


class MultiLayerModule(nn.Module):
    '''
    Multi Layer perceptron.
    '''

    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_shapes,
                 activation,
                 output_layer_name='output_layer'):

        super(MultiLayerModule, self).__init__()

        self.module = self.make_module(input_shape,
                                       output_shape,
                                       hidden_shapes,
                                       activation,
                                       output_layer_name
                                       )
        return None

    def make_module(self, input_shape, output_shape, hidden_shapes, activation, output_layer_name):
        module = OrderedDict({})
        shapes = [input_shape] + hidden_shapes

        for i in range(len(shapes) - 1):
            module.update(dense_norm_layer(i + 1, shapes[i], shapes[i + 1], activation))

        module.update(OrderedDict({output_layer_name: nn.Linear(shapes[-1], output_shape)}))

        return nn.Sequential(module)

    def forward(self, z, c):
        inputs = torch.cat([z, c], -1)
        output = self.module(inputs)
        return output


class VariationnalModel(pl.LightningModule):
    '''
    A semi supervised variationnal model. It handles both continuous and categorical data types simultaneously.

    Caveat: the categorical dimensions must all have the same number of classes (eg binary pixels, SNPs, ...) 

    Parameters:
    -----------
    latent_shape: int
        Dimension of the latent space
    activation: nn.Module()
        Activation function of the hidden layers
    hidden_shapes: list of int
        List of shapes of the hidden layers
    X_shape: int
        Input shape
    C_shape: int
        Shape of conditionnal input
    X_continuous_shape: int
        Number of continous dimensions in the input vector
    X_categorical_shape: int
        Number of categorical dimensions in the input vector
    X_categorical_weight: int
        Prior weights for the reconstruction classification problem
    X_categorical_n_classes:int
        Number of classes for the categorical dimensions in the input vector
    y_continuous_shape: int
        Number of continuous dimensions in the target vector
    y_categorical_shape: int
        Number of categorical dimensions in the target vector
    y_categorical_n_classes: int
        Number of classes for the categorical dimensions in the target vector
    y_categorical_labels: None
        Deprecated
    y_categorical_labels_name: None
        Deprecated
    learning_rate: float
        Learning rate for the optimizer
    lambda_l1: float
        L1 regularization coefficient (encoder only)
    lambda_l2: float
        L2 regularization coefficient (encoder only)
    alpha: float
        Coefficient of the reconstruction loss
    beta: float
        Coefficient of the encoding loss
    gamma: float
        Coefficient of the supervised loss
    '''

    def __init__(self,
                 latent_shape=10,
                 activation=nn.Identity(),  # Identity
                 hidden_shapes=[],
                 X_shape=None,
                 C_shape=0,
                 X_continuous_shape=0,
                 X_categorical_shape=0,
                 X_categorical_weight=None,
                 X_categorical_n_classes=0,
                 y_continuous_shape=0,
                 y_categorical_shape=0,
                 y_categorical_weight=None,
                 y_categorical_n_classes=0,
                 y_categorical_labels=None,
                 y_categorical_labels_name='auto',
                 learning_rate=1e-3,
                 weight_decay=0,
                 lambda_l1=0,
                 lambda_l2=0,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 ):

        super(VariationnalModel, self).__init__()
        self.save_hyperparameters()

        assert X_shape == X_categorical_shape + \
            X_continuous_shape, 'X_categorical_shape + X_continuous_shape must equal X_shape'

        # shapes
        self.latent_shape = latent_shape

        self.X_continuous_shape = X_continuous_shape
        self.X_categorical_shape = X_categorical_shape
        self.X_categorical_n_classes = X_categorical_n_classes

        self.y_continuous_shape = y_continuous_shape
        self.y_categorical_shape = y_categorical_shape
        self.y_categorical_n_classes = y_categorical_n_classes

        self.x_split_sizes = [self.X_continuous_shape, self.X_continuous_shape,
                              self.X_categorical_shape * self.X_categorical_n_classes]

        self.encoder_split_sizes = [self.latent_shape, self.latent_shape, self.y_continuous_shape,
                                    self.y_continuous_shape, self.y_categorical_shape * self.y_categorical_n_classes]

        encoder_input_shape = X_shape + C_shape
        encoder_output_shape = 2 * latent_shape + 2 * y_continuous_shape + \
            y_categorical_shape * y_categorical_n_classes
        encoder_output_layer_name = '(z_mean_logvar)_(y_mean_logvar)_(classification_logits)'

        decoder_input_shape = C_shape + latent_shape + y_continuous_shape + \
            y_categorical_shape * y_categorical_n_classes
        decoder_output_shape = 2 * X_continuous_shape + X_categorical_shape * X_categorical_n_classes
        decoder_output_layer_name = '(X_continuous_mean_logvar)_(X_categorical_logits)'

        # encoder and decode
        self.encoder = MultiLayerModule(encoder_input_shape,
                                        encoder_output_shape,
                                        hidden_shapes,
                                        activation,
                                        encoder_output_layer_name)

        self.decoder = MultiLayerModule(decoder_input_shape,
                                        decoder_output_shape,
                                        hidden_shapes[::-1],
                                        activation,
                                        decoder_output_layer_name)

        # distributions
        self.latent_distribution = MultivariateNormal(
            torch.zeros(latent_shape), torch.eye(latent_shape))

        if self.y_continuous_shape != 0:
            self.y_distribution = MultivariateNormal(
                torch.zeros(y_continuous_shape), torch.eye(y_continuous_shape))

        if self.X_continuous_shape != 0:
            self.X_distribution = MultivariateNormal(
                torch.zeros(X_continuous_shape), torch.eye(X_continuous_shape))

        # loss coefficients
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
        self.X_categorical_weight = X_categorical_weight
        self.y_categorical_weight = y_categorical_weight

        # Plots
        self.y_categorical_labels = y_categorical_labels
        self.y_categorical_labels_name = y_categorical_labels_name

    @staticmethod
    def add_model_specific_args(parent_parser):
        '''
        Add alpha, beta, gamma
        '''
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent_shape', type=int, default=12)
        parser.add_argument('--hidden_shapes', type=int, nargs='+', default=[])
        return parser

    def encode(self, x, c):
        encoder_output = self.encoder(x, c)

        z_mu, z_logvar, y_mu, y_logvar, y_logits = torch.split(
            encoder_output, self.encoder_split_sizes, dim=1)

        output = {'z_mu': z_mu, 'z_logvar': z_logvar, 'y_mu': y_mu,
                  'y_logvar': y_logvar, 'y_logits': y_logits}

        return output

    def decode(self, z, c):
        decoder_output = self.decoder(z, c)

        X_mu, X_logvar, X_logits = torch.split(decoder_output, self.x_split_sizes, dim=1)

        if self.X_categorical_shape != 0:
            X_logits = X_logits.reshape(-1, self.X_categorical_n_classes, self.X_categorical_shape)

        output = {'X_mu': X_mu, 'X_logvar': X_logvar, 'X_logits': X_logits}

        return output

    def forward(self, x, c, n=1):

        encoder_outputs = self.encode(x, c)

        z = self.reparameterize(
            encoder_outputs['z_mu'], encoder_outputs['z_logvar'], n, self.latent_distribution)
        c = c.repeat(n, 1)

        decoder_input = z

        if self.y_continuous_shape != 0:
            y_continuous = self.reparameterize(
                encoder_outputs['y_mu'], encoder_outputs['y_logvar'], n, self.y_distribution)
            decoder_input = torch.cat([decoder_input, y_continuous], dim=1)

        y_probas = nn.Softmax(dim=1)(encoder_outputs['y_logits'].repeat(n, 1))

        decoder_input = torch.cat([decoder_input, y_probas], dim=1)

        decoder_outputs = self.decode(decoder_input, c)

        outputs = {
            "encoder": encoder_outputs,
            "decoder": decoder_outputs,
            'z':z
        }

        return outputs

    def reparameterize(self, mu, logvar, n, distribution):
        mu = mu.repeat(n, 1)
        logvar = logvar.repeat(n, 1)
        std = torch.exp(0.5 * logvar)
        eps = distribution.sample(sample_shape=(mu.shape[0],)).type_as(std)
        z = mu + std * eps
        return z

    def compute_loss_continuous(self, x_continuous, x_mu, x_logvar):
        if x_continuous.shape[-1] != 0:
            return gaussian_nll(x_continuous, x_mu, x_logvar)
        else:
            return 0

    def compute_loss_categorical(self, x_categorical, x_logits, weight=None):

        if x_categorical.shape[-1] != 0:
            return F.cross_entropy(x_logits, x_categorical.long(), reduction='mean', weight=weight)
        else:
            return 0

    def format_categorical_output(self, var_categorical, var_logits, var_name):
        if var_categorical.shape[-1] != 0:
            return {f'{var_name}_categorical_pred': nn.Softmax(dim=1)(var_logits).argmax(dim=1), f'{var_name}_categorical': var_categorical}
        else:
            return {f'{var_name}_categorical_pred': None, f'{var_name}_categorical': None}

    def format_continuous_output(self, var_continuous, var_mu, var_name):
        if var_continuous.shape[-1] != 0:
            return {f'{var_name}_continuous_pred': var_mu, f'{var_name}_continuous': var_continuous}
        else:
            return {f'{var_name}_continuous_pred': None, f'{var_name}_continuous': None}

    def _step(self, batch, batch_idx, step_name):
        '''
        The model optimizes simultaneously the sum of three objectives.
        As in the VAE, negative log likehood (nll) of the ouput and Kulback Leibler divergence of the latent representation is computed.
        Additionnaly, a classification loss is added to classify the input in a given category.
        The coefficients alpha, beta, gamma are used for tuning the model.
        To do:
        - Clean up, decide wether or not use one hot for encoder
        - Only return loss, return other stuff only at validation
        FIND a nice way to deal with the ifs
        '''
        x, c, y = batch[0], batch[1], batch[2]

        n = 1

        model_outputs = self.forward(x, c, n)

        x_continuous, x_categorical = torch.split(
            x, [self.X_continuous_shape, self.X_categorical_shape], dim=1)
        y_continuous, y_categorical = torch.split(
            y, [self.y_continuous_shape, self.y_categorical_shape], dim=1)

        # define losses

        # Latent space loss
        kld = normal_kld(mu=model_outputs['encoder']['z_mu'],
                         logvar=model_outputs['encoder']['z_logvar'])

        # reconstruction losses
        nll_continuous = self.compute_loss_continuous(
            x_continuous, model_outputs['decoder']['X_mu'], model_outputs['decoder']['X_logvar'])

        nll_categorical = self.compute_loss_categorical(
            x_categorical, model_outputs['decoder']['X_logits'], weight=self.X_categorical_weight)

        recon_loss = nll_continuous + nll_categorical

        # supervised losses
        regression_loss = self.compute_loss_continuous(
            y_continuous, model_outputs['encoder']['y_mu'], model_outputs['encoder']['y_logvar'])

        classification_loss = self.compute_loss_categorical(
            y_categorical.flatten(), model_outputs['encoder']['y_logits'], weight=self.y_categorical_weight)

        supervised_loss = regression_loss + classification_loss

        # compute total loss
        loss = self.alpha * recon_loss + self.beta * kld + self.gamma * supervised_loss
        
        loss += self.lambda_l1 * l1_reg(self.encoder) + self.lambda_l2 * l2_reg(self.encoder)

        step_outputs = {}
        step_outputs.update({'loss': loss})

        # logs
        self.log(f'Loss/{step_name}_loss', loss)
        self.log(f'KLD/{step_name}_kld', kld)

        self.log(f'Recon/{step_name}_nll_continuous', nll_continuous)
        self.log(f'Recon/{step_name}_nll_categorical', nll_categorical)

        self.log(f'Supervised/{step_name}_regression_loss', regression_loss)
        self.log(f'Supervised/{step_name}_classification_loss', classification_loss)

        if step_name == 'validation':

            step_outputs.update({'mu_latent': model_outputs['z'].cpu()})

            step_outputs.update(self.format_continuous_output(
                x_continuous.cpu(), model_outputs['decoder']['X_mu'].cpu(), 'x'))
            step_outputs.update(self.format_continuous_output(
                y_continuous.cpu(), model_outputs['encoder']['y_mu'].cpu(), 'y'))

            step_outputs.update(self.format_categorical_output(
                x_categorical.cpu(), model_outputs['decoder']['X_logits'].cpu(), 'x'))
            step_outputs.update(self.format_categorical_output(
                y_categorical.cpu(), model_outputs['encoder']['y_logits'].cpu(), 'y'))

        return step_outputs

    def training_step(self, batch, batch_idx):
        step_results = self._step(batch, batch_idx, step_name='train')
        return step_results['loss']

    def validation_step(self, batch, batch_idx):
        '''
        Add metrics here !
        '''
        step_results = self._step(batch, batch_idx, step_name='validation')
        return step_results

    def test_step(self, batch, batch_idx):
        step_results = self._step(batch, batch_idx, step_name='test')
        return step_results['loss']

    def validation_epoch_end(self, outputs):
        '''To be improved:
            make plotting faster (less/smaller plots)
            sample subset to make the plot to reduce mem usage using trainer
            Do batchwise validation as data may become large
        '''
        outputs = outputs[0]
        del outputs['loss']
        diagnostic_plots = make_diagnostic_plots(**outputs)

        for plot_name, plot in diagnostic_plots.items():
            self.logger.experiment.add_figure(tag=plot_name,
                                              figure=plot,
                                              global_step=self.current_epoch)

    def training_epoch_end(self, outputs):
        """TO DO"""

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)
        return optimizer

