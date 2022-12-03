# %% Imports

import os
os.chdir('../ssl_neuron/')
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from allensdk.core.cell_types_cache import CellTypesCache

from ssl_neuron.datasets import AllenDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import time

# %% Setup
"""
config = json.load(open('./ssl_neuron/configs/config.json'))
config['data']['n_nodes'] = 1000

ctc = CellTypesCache(manifest_file='./ssl_neuron/data/cell_types/manifest.json')

cells = ctc.get_cells()

ephys_features = ctc.get_ephys_features()
ef_df = pd.DataFrame(ephys_features)

morphology_features = ctc.get_morphology_features()
morph_df = pd.DataFrame(morphology_features)

cell_df = pd.DataFrame(cells)

dset = AllenDataset(config, mode='all')

latents = np.load('../analysis/latents.npy')

ef_df = ef_df.set_index('specimen_id').loc[dset.cell_ids]
morph_df = morph_df[~morph_df.superseded].set_index('specimen_id').loc[dset.cell_ids]
cell_df = cell_df.set_index('id').loc[dset.cell_ids]

# %% Prep features

cell_type_input_columns = ['reporter_status',
       'structure_layer_name', 'structure_area_id', 'structure_area_abbrev',
       'transgenic_line', 
       'dendrite_type', 'apical', 'reconstruction_type',
       'structure_hemisphere',
       'normalized_depth']
cell_features = []
for column in cell_type_input_columns:
    data = cell_df[column]
    if column == 'normalized_depth':
        data = data.to_numpy()
        data = StandardScaler().fit_transform(data[:, None])
    elif column == 'cell_soma_location':
        data = np.array(data.tolist())
        data = StandardScaler().fit_transform(data)
    else:
        if column == 'structure_area_id':
            data = np.array([str(sa_id) for sa_id in data], dtype='object')
        else:
            data = data.to_numpy()
        data = OneHotEncoder().fit_transform(data[:, None]).todense()
    cell_features.append(data)
cell_features = np.concatenate(cell_features, axis=1)

morph_input_columns = ['average_bifurcation_angle_local',
       'average_contraction', 'average_diameter', 'average_fragmentation',
       'average_parent_daughter_ratio',
       'max_branch_order', 'max_euclidean_distance', 'max_path_distance',
       'number_bifurcations', 'number_branches',
       'number_nodes', 'number_stems', 'number_tips', 'overall_depth',
       'overall_height', 'overall_width', 'soma_surface', 'total_length',
       'total_surface', 'total_volume']
morph_features = []
for column in morph_input_columns:
    data = morph_df[column]
    data = data.to_numpy()
    data = StandardScaler().fit_transform(data[:, None])
    morph_features.append(data)
morph_features = np.concatenate(morph_features, axis=1)

# %% Define function

def fit_eval_decoder(input_features, target_df, skip_cols=[], to_str_cols=[], 
                     regression_model=Ridge, regression_params={'alpha': np.logspace(-8, 3, 12)},
                     classification_model=LogisticRegression, classification_params={},
                     seed=0):
    np.random.seed(seed)
    score_dict = {}
    score_std_dict = {}
    pred_truth_dict = {}
    for col in target_df.columns:
        if col in skip_cols:
            continue
        elif col in to_str_cols:
            targets = np.array([str(item) for item in target_df[col]], dtype='object')
        else:
            targets = target_df[col].to_numpy()
        if targets.dtype == float or targets.dtype == int:
            gscv = GridSearchCV(regression_model(), regression_params)
            # gscv = GridSearchCV(SVR(), {'C': np.logspace(-8, 0, 5)})
            mask = np.isnan(targets)
            if np.sum(~mask) == 0:
                print(f'skipping col {col} because there is no valid data')
                continue
            inputs = input_features[~mask]
            targets = targets[~mask]
            targets = StandardScaler().fit_transform(targets[:, None]).flatten()
        elif targets.dtype == bool:
            if len(np.unique(targets)) < 2:
                print(f'skipping col {col} because there is only one value')
                continue
            inputs = input_features
            targets = targets.astype(int)
            gscv = GridSearchCV(classification_model(), classification_params)
        elif type(targets[0]) == str:
            if len(np.unique(targets)) < 2:
                print(f'skipping col {col} because there is only one value')
                continue
            inputs = input_features
            targets = LabelEncoder().fit_transform(targets)
            gscv = GridSearchCV(classification_model(), classification_params)
        else:
            print(f'skipping col {col} due to unsupported dtype {targets.dtype}')
            continue
        perm = np.random.permutation(inputs.shape[0])
        gscv.fit(inputs[perm], targets[perm])
        score_dict[col] = gscv.best_score_
        score_std_dict[col] = gscv.cv_results_['std_test_score'][gscv.best_index_]
        # escore_dict[col] = gscv.score(inputs, targets)
        pred_truth_dict[col] = (gscv.predict(inputs), targets)
    return score_dict, score_std_dict, pred_truth_dict
"""

"""
# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, nonlinearity='relu', output_activation='none', dropout=0.1):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(
                (input_size if i == 0 else layer_sizes[i - 1]),
                (output_size if i == len(layer_sizes) else layer_sizes[i]),
                bias=True
            ) for i in range(len(layer_sizes) + 1)
        ])
        if nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        else:
            self.nonlinearity = getattr(F, nonlinearity) if nonlinearity != 'none' else lambda x: x
        self.output_activation = getattr(F, output_activation) if output_activation != 'none' else lambda x: x
        self.dropout_rate = dropout
    
    def forward(self, X):
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if i == len(self.layers) - 1:
                X = self.output_activation(X)
            else:
                X = self.nonlinearity(X)
                X = F.dropout(X, p=self.dropout_rate, training=self.training)
        return X
    
    def set_dropout(self, dropout):
        self.dropout_rate = dropout

class MLPEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, layer_sizes=[8], nonlinearity='relu', 
                 output_activation='none', dropout=0.1, weight_decay=0., max_iters=2000, patience=200):
        super(MLPEstimator, self).__init__()
        # self.model = MLP(
        #     input_size=input_size, layer_sizes=layer_sizes, output_size=output_size, nonlinearity=nonlinearity,
        #     output_activation=output_activation, dropout=dropout
        # )
        self.model = None
        self.max_iters = max_iters
        self.patience = patience
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.layer_sizes = layer_sizes
        self.nonlinearity = nonlinearity
        self.output_activation = output_activation
    
    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]
        if True: # self.model is None:
            self.model = MLP(X.shape[1], self.layer_sizes, y.shape[1], self.nonlinearity, self.output_activation, self.dropout)
        X = torch.from_numpy(X).to(torch.float)
        y = torch.from_numpy(y).to(torch.float)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

        best_loss = np.inf
        last_improv = 0
        for i in range(self.max_iters):
            optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                last_improv = 0
            else:
                last_improv += 1
            
            if last_improv > self.patience:
                break
        self.best_loss = best_loss
        self.final_loss = loss.item()
        # print(self.best_loss, self.final_loss)

    def predict(self, X):
        if self.model is None:
            raise AssertionError("Model not fit yet")
        X = torch.from_numpy(X).to(torch.float)
        y = self.model(X)
        y = y.detach().cpu().numpy()
        return y

    def get_params(self, deep=True):
        return {
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "layer_sizes": self.layer_sizes,
            "nonlinearity": self.nonlinearity,
            "output_activation": self.output_activation,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == 'dropout':
                self.dropout = value
                # self.model.set_dropout(value)
            elif parameter == 'weight_decay':
                self.weight_decay = value
        return self

# %% Latents only

print('latents')
print(time.time())

latent_features = StandardScaler().fit_transform(latents)

skip_cols = ['electrode_0_pa', 'has_burst', 'has_delay', 'has_pause', 'id', 'rheobase_sweep_id', 'rheobase_sweep_number', 'vm_for_sag']

llscore_dict, llscore_std_dict, llpred_truth_dict = fit_eval_decoder(
    latent_features, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

lnscore_dict, lnscore_std_dict, lnpred_truth_dict = fit_eval_decoder(
    latent_features, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [llscore_dict, llscore_std_dict, lnscore_dict, lnscore_std_dict]:
    d['input_features'] = 'latents'

# lcscore_dict, lcscore_std_dict, lcpred_truth_dict = fit_eval_decoder(
#     latents, ef_df, skip_cols=skip_cols, 
#     regression_model=MLPEstimator, 
#     regression_params={'weight_decay': np.logspace(-6, 4, 2), 'dropout': np.linspace(0, 0.2, 2)},
#     classification_model=LogisticRegression,
#     seed=0
# )

# print(lcscore_dict)

# import pdb; pdb.set_trace()

# %% Cell type only

print('cell type')
print(time.time())

clscore_dict, clscore_std_dict, clpred_truth_dict = fit_eval_decoder(
    cell_features, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

cnscore_dict, cnscore_std_dict, cnpred_truth_dict = fit_eval_decoder(
    cell_features, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [clscore_dict, clscore_std_dict, cnscore_dict, cnscore_std_dict]:
    d['input_features'] = 'cell'

# %% Morph only

print('morphology')
print(time.time())

mlscore_dict, mlscore_std_dict, mlpred_truth_dict = fit_eval_decoder(
    morph_features, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

mnscore_dict, mnscore_std_dict, mnpred_truth_dict = fit_eval_decoder(
    morph_features, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [mlscore_dict, mlscore_std_dict, mnscore_dict, mnscore_std_dict]:
    d['input_features'] = 'morph'

# %% Latents + Cell type

print('latents + cell type')
print(time.time())

inputs = np.concatenate([latent_features, cell_features], axis=1)

lclscore_dict, lclscore_std_dict, lclpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

lcnscore_dict, lcnscore_std_dict, lcnpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [lclscore_dict, lclscore_std_dict, lcnscore_dict, lcnscore_std_dict]:
    d['input_features'] = 'latents+cell'

# %% Latents + morph

print('latents + morph')
print(time.time())

inputs = np.concatenate([latent_features, morph_features], axis=1)

lmlscore_dict, lmlscore_std_dict, lmlpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

lmnscore_dict, lmnscore_std_dict, lmnpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [lmlscore_dict, lmlscore_std_dict, lmnscore_dict, lmnscore_std_dict]:
    d['input_features'] = 'latents+morph'

# %% Cell type + morph

print('cell type + morph')
print(time.time())

inputs = np.concatenate([cell_features, morph_features], axis=1)

cmlscore_dict, cmlscore_std_dict, cmlpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

cmnscore_dict, cmnscore_std_dict, cmnpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [cmlscore_dict, cmlscore_std_dict, cmnscore_dict, cmnscore_std_dict]:
    d['input_features'] = 'cell+morph'

# %% Latents + cell types + morph

print('all three')
print(time.time())

inputs = np.concatenate([latent_features, cell_features, morph_features], axis=1)

lcmlscore_dict, lcmlscore_std_dict, lcmlpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    seed=0
)

lcmnscore_dict, lcmnscore_std_dict, lcmnpred_truth_dict = fit_eval_decoder(
    inputs, ef_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

for d in [lcmlscore_dict, lcmlscore_std_dict, lcmnscore_dict, lcmnscore_std_dict]:
    d['input_features'] = 'latents+cell+morph'

# %%

print(time.time())
# import pdb; pdb.set_trace()

linear_scores = [llscore_dict, clscore_dict, mlscore_dict, lclscore_dict, lmlscore_dict, cmlscore_dict, lcmlscore_dict]
linear_scores = pd.DataFrame(linear_scores)
linear_scores.to_csv('../analysis/linear_scores.csv')

linear_score_stds = [llscore_std_dict, clscore_std_dict, mlscore_std_dict, lclscore_std_dict, lmlscore_std_dict, cmlscore_std_dict, lcmlscore_std_dict]
linear_score_stds = pd.DataFrame(linear_score_stds)
linear_score_stds.to_csv('../analysis/linear_score_stds.csv')

nonlinear_scores = [lnscore_dict, cnscore_dict, mnscore_dict, lcnscore_dict, lmnscore_dict, cmnscore_dict, lcmnscore_dict]
nonlinear_scores = pd.DataFrame(nonlinear_scores)
nonlinear_scores.to_csv('../analysis/nonlinear_scores.csv')

nonlinear_score_stds = [lnscore_std_dict, cnscore_std_dict, mnscore_std_dict, lcnscore_std_dict, lmnscore_std_dict, cmnscore_std_dict, lcmnscore_std_dict]
nonlinear_score_stds = pd.DataFrame(nonlinear_score_stds)
nonlinear_score_stds.to_csv('../analysis/nonlinear_score_stds.csv')
"""

# %%

linear_scores = pd.read_csv('../analysis/linear_scores.csv')
linear_score_stds = pd.read_csv('../analysis/linear_score_stds.csv')
nonlinear_scores = pd.read_csv('../analysis/nonlinear_scores.csv')
nonlinear_score_stds = pd.read_csv('../analysis/nonlinear_score_stds.csv')

# %%

for feature in linear_scores.columns:
    if feature == 'input_features':
        continue

    fig, axs = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    
    axs[0].bar(np.arange(7), linear_scores[feature], yerr=(linear_score_stds[feature] / np.sqrt(5)))
    axs[0].set_xticks(np.arange(7))
    axs[0].set_xticklabels(linear_scores['input_features'], rotation=90)
    axs[0].set_title('Linear')
    
    axs[1].bar(np.arange(7), nonlinear_scores[feature], yerr=(nonlinear_score_stds[feature] / np.sqrt(5)))
    axs[1].set_xticks(np.arange(7))
    axs[1].set_xticklabels(nonlinear_scores['input_features'], rotation=90)
    axs[1].set_title('Non-linear')

    plt.tight_layout()
    plt.savefig(f'../analysis/score_plots/{feature}.png')
    plt.close()

