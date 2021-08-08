import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score
import plotly.express as px
import plotly.graph_objects as go
import torch

plt.style.use('ggplot')

# Tensorboard plots

def embeddings_scatterplot(df):
    '''To improve
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(data=df, x="dim_1", y="dim_2", hue="label", ax=ax[0])
    sns.scatterplot(data=df, x="dim_1", y="dim_3", hue="label", ax=ax[1])
    plt.close()
    return fig


def confusion_matrix_heatmap(y_true, y_pred, normalize, labels=None, labels_name="auto"):
    ''''''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred,
                          normalize=normalize, labels=labels)
    sns.heatmap(cm, annot=True, ax=ax, xticklabels=labels_name, yticklabels=labels_name)
    plt.close()
    return fig


def covariance_matrix_heatmap(df):
    ''''''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(df.cov(), ax=ax)
    plt.close()
    return fig


def embedding_distributions(df):
    ''''''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.violinplot(x="variable", y="value", data=df, ax=ax)
    plt.close()
    return fig


def make_latent_space_dataframe(mu_latent, n_samples=1000):
    ''''''
    df = pd.DataFrame(mu_latent)[0:n_samples]
    dim_names = [f"dim_{i + 1}" for i in range(mu_latent.shape[1])]
    col_names = dim_names
    df.columns = col_names
    return df


def regression_scatterplot(y_true, y_pred):
    ''''''
    r2 = r2_score(y_true, y_pred)
    
    g = sns.jointplot(y=y_pred, x=y_true, kind='scatter',alpha=0.1, height=10)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.title(f'r2: {r2}')
    plt.close(g.fig)
    return g.fig


def make_diagnostic_plots(x_categorical=None,
                          x_categorical_pred=None,
                          x_continuous=None,
                          x_continuous_pred=None,
                          y_continuous=None,
                          y_continuous_pred=None,
                          y_categorical=None,
                          y_categorical_pred=None,
                          mu_latent=None,
                          y_labels=None,
                          y_labels_name="auto"):
    """Return a dictionnary of diagnostic plots to be displayed in tensorboard.
    The visualizations allow for easy debbuging and sanity checking of the model.
    Embeddings/Covariance displays the covariance matrix of the latent vector to check for the isotropic assumption.
    Embeddings/Distributions displays distribution of the latent space to check for gaussian assumption.
    Classification/Confusion Matrix displays the confusion matrix for the classification problem.
    Regression/Regression Diagnostic displays the diagnostics for the regression problem
    Reconstruction/Confusion Matrix for the reconstruction problem.
    Reconstruction/Regression Diagnostic displays diagnostics for the reconstruction regression problem
    """
    df = make_latent_space_dataframe(mu_latent.detach().numpy(), n_samples=1000)

    plots = {}

    plots.update({'Embeddings/Covariance': covariance_matrix_heatmap(df)})

    plots.update({'Embeddings/Distributions': embedding_distributions(df.melt())})

    if y_categorical_pred is not None:
        plots.update({'Classification/Confusion Matrix': confusion_matrix_heatmap(y_categorical.detach().flatten(),
                                                                                  y_categorical_pred.detach().flatten(), normalize=None, labels=y_labels, labels_name=y_labels_name)})

    if y_continuous_pred is not None:
        plots.update({'Regression/Regression Diagnostic': regression_scatterplot(
            y_continuous.detach().flatten(), y_continuous_pred.detach().flatten())})

    if x_categorical_pred is not None:
        plots.update({'Reconstruction/Confusion Matrix': confusion_matrix_heatmap(
            x_categorical.detach().flatten(), x_categorical_pred.detach().flatten(), normalize='true')})

    if x_continuous_pred is not None:
        plots.update({'Reconstruction/Regression Diagnostic': regression_scatterplot(
            x_continuous.detach().flatten(), x_continuous_pred.detach().flatten())})

    return plots



# Ternary plot

# Auxiliary df for clean plotly plots

def sample_i_df(probas, i):
    return pd.DataFrame(probas[i], columns=['P0', 'P1', 'P2'])\
             .assign(sample_id=i)\
             .reset_index().rename(columns={'index': 'SNP_id'})


def reconstruction_snp_df(probas, true_SNP):
    df = pd.concat([sample_i_df(probas, i) for i in range(probas.shape[0])])

    df['true_SNP'] = true_SNP
    df['true_SNP'] = df['true_SNP'].astype(int).astype(str)

    return df

# Plots


def boundary(fig, a, b, c):
    return fig.add_trace(go.Scatterternary({
        'mode': 'lines',
        'a': a,
        'b': b,
        'c': c,
        'line': {'width': 2, 'dash': 'dash', 'color': 'darkgray'},
        'showlegend': False
    }))


def add_boudaries(fig):
    for i in range(3):
        a_b_c = [[1 / 2, 1 / 3]] * 3
        a_b_c[i] = [0, 1 / 3]
        fig = boundary(fig, *a_b_c)
    return fig


def plot_reconstruction_on_simplex(df):
    fig = px.scatter_ternary(df,
                             a='P0',
                             b='P1',
                             c='P2',
                             color='true_SNP',
                             hover_data=['SNP_id', 'sample_id'],
                             color_discrete_map={'0': 'blue', '1': 'red', '2': 'green'},
                             )

    fig.update_ternaries(baxis_dtick=0.5, aaxis_dtick=0.5, caxis_dtick=0.5)

    return add_boudaries(fig)


# Sampling plots

def sample_snp_df(probas):
    if isinstance(probas, torch.Tensor):
        probas = probas.detach().numpy()

    df = pd.concat([sample_i_df(probas, i) for i in range(probas.shape[0])])

    return df


def sample_snp_y_df(probas):
    df = pd.concat([sample_snp_df(proba).assign(y=str(i)) for i, proba in enumerate(probas)])
    return df


def plot_samples_on_simplex(df):
    fig = px.scatter_ternary(df,
                             a='P0',
                             b='P1',
                             c='P2',
                             color='y',
                             hover_data=['SNP_id', 'sample_id'],
                             opacity=0.6,
                             #  color_discrete_map={'0': 'blue', '1': 'red', '2': 'green'},
                             )
    fig.update_ternaries(baxis_dtick=0.5, aaxis_dtick=0.5, caxis_dtick=0.5)

    return add_boudaries(fig)



# SNP sampling plots
def get_frequencies_by_SNP(X):
    '''To improve'''
    freqs = []
    for i in range(3):
        freqs.append((X == i).sum(axis=0))
    return np.array(freqs) / X.shape[0]


def make_freq_df(X):
    frequencies_by_dim = get_frequencies_by_SNP(X)
    df = (
        pd.DataFrame(frequencies_by_dim, columns=X.columns)
          .assign(SNP=[0, 1, 2])
          .melt(id_vars=['SNP'], var_name='rsid')
    )
    return df


def plot_allele_freq(X_A, X_B, A_name='test_set', B_name='sample_set', lib='plotly'):
    
    df_A = make_freq_df(X_A).assign(sampled_from=A_name)
    df_B = make_freq_df(X_B).assign(sampled_from=B_name)
    
    plot_df = pd.concat([df_A, df_B])\
        .pivot(index=['SNP', 'rsid'], columns=['sampled_from'], values='value')\
        .reset_index()\
        .astype({'SNP': str}
               )
    
    if lib == 'plotly':
        fig = px.scatter(plot_df, x=A_name,
                               y=B_name, 
                               color='SNP',
                               opacity=0.4,
                               hover_data=['rsid'],
                               title=f'SNP frequencies: {A_name} vs {B_name}',
                               marginal_x="box", marginal_y="box",
                               height=700,
                               width=700)
        return fig

        
    if lib=='sns':
        fig, ax = plt.subplots(1, 1, figsize=(5 ,5))
        fig = sns.scatterplot(data=plot_df, x=A_name,
                       y=B_name, 
                       hue='SNP', ax=ax, alpha=0.5)
        plt.show()
        

def make_SNP_corr_df(X):
    corr = np.corrcoef(X.T)
    df = (
        pd.DataFrame(corr, columns=X.columns, index=X.columns)
          .reset_index()
          .rename(columns={'index': 'rsid_1'})
          .melt(id_vars=['rsid_1'], var_name='rsid_2')
    )
    return df[df.rsid_1 != df.rsid_2] # removes pairs of same SNP


def plot_SNP_corr(X_A, X_B, A_name='test_set', B_name='sample_set'):
    df_sample = make_SNP_corr_df(X_A).assign(sampled_from=A_name)
    df_test = make_SNP_corr_df(X_B).assign(sampled_from=B_name)

    df = (
        pd.concat([df_sample, df_test])
          .pivot(index=['rsid_1', 'rsid_2'], columns='sampled_from', values='value')\
          .reset_index()\
    )

    fig = px.scatter(df,
                 y=A_name,
                 x=B_name,
                 hover_data=['rsid_1', 'rsid_2'],
                 height=700,
                 width=700)

    fig.update_yaxes(scaleanchor = "x",
                     scaleratio = 1,
                    )
    return fig

