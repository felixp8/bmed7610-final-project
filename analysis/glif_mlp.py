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
from sklearn.svm import SVR, SVC

from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities

import time

# %% Setup

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

latent_features = StandardScaler().fit_transform(latents)

cell_type_input_columns = ['reporter_status',
       'structure_layer_name', 'structure_area_id', 'structure_area_abbrev',
       'transgenic_line', 
       'dendrite_type', 'apical', 'reconstruction_type',
       'structure_hemisphere',
       'normalized_depth']
cell_features = []
cell_feature_names = []
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
    cell_feature_names += [column] * data.shape[1]
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
morph_feature_names = morph_input_columns.copy()
for column in morph_input_columns:
    data = morph_df[column]
    data = data.to_numpy()
    data = StandardScaler().fit_transform(data[:, None])
    morph_features.append(data)
morph_features = np.concatenate(morph_features, axis=1)

# %% Prep target

glif_api = GlifApi()

model_ids = []
model_vars = []
has_model = []
cell_ids = []
for cell_id in dset.cell_ids:
    nm = glif_api.get_neuronal_models(cell_id)
    if len(nm) < 1:
        # print(f'{cell_id}*, ', end='')
        has_model.append(False)
        continue
    nm = nm[0]['neuronal_models']
    model_id = None
    for model in nm:
        if '3' in model['name'][:2]: # get basic LIF neurons
            model_id = model['id']
            try:
                var = model['neuronal_model_runs'][0]['explained_variance_ratio']
            except:
                var = None
            break
    if model_id is None:
        # print(f'{cell_id}-, ', end='')
        has_model.append(False)
        continue
    model_ids.append(model_id)
    has_model.append(True)
    cell_ids.append(cell_id)
    model_vars.append(var)

configs = glif_api.get_neuron_configs(model_ids)

def flatten_dict(dictionary, prefix=''):
    flattened = {}
    for key, value in dictionary.items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                flattened[f'{prefix}{key}_{i}'] = item
        elif isinstance(value, dict):
            if len(value) > 0:
                flattened.update(flatten_dict(value, key + '_'))
        else:
            flattened[f'{prefix}{key}'] = value
    return flattened
model_data = []
for cell_id, model_id in zip(cell_ids, model_ids):
    model_config = configs[model_id]
    keep_config = flatten_dict(model_config)
    keep_config['model_id'] = model_id
    keep_config['cell_id'] = cell_id
    model_data.append(keep_config)

model_df = pd.DataFrame(model_data)

for column in model_df.columns:
    if model_df[column].nunique() <= 1:
        model_df.drop(column, axis=1, inplace=True)
if np.max(np.abs(model_df['th_inf'] - model_df['init_threshold'])) < 1e-6:
    model_df.drop('th_inf', axis=1, inplace=True)
print(model_df.columns)

has_model = np.array(has_model)
print(has_model.sum())

# %% Define function

# ['El_reference', 'C', 'asc_amp_array_0', 'asc_amp_array_1',
    #    'init_threshold', 'spike_cut_length', 'asc_tau_array_0',
    #    'asc_tau_array_1', 'R_input', 'coeffs_th_inf', 'model_id', 'cell_id']

def fit_eval_decoder(input_features, full_input_features,
                     regression_model=Ridge, regression_params={'alpha': np.logspace(-8, 3, 12)},
                     classification_model=LogisticRegression, classification_params={},
                     seed=0, return_models=False):
    np.random.seed(seed)
    score_dict = {}
    score_std_dict = {}
    pred_truth_dict = {}
    model_dict = {}
    for col in model_df.columns:
        if col in ['model_id', 'cell_id']:
            continue
        else:
            targets = model_df[col].to_numpy()
        if col in ['El_reference', 'C', 'asc_amp_array_0', 'asc_amp_array_1', 'init_threshold', 
            'spike_cut_length', 'R_input', 'coeffs_th_inf']:

            gscv = GridSearchCV(regression_model(), regression_params)
            # gscv = GridSearchCV(SVR(), {'C': np.logspace(-8, 0, 5)})
            inputs = input_features
            targets = targets
            scaler = StandardScaler()
            targets = scaler.fit_transform(targets[:, None]).flatten()
        elif col in ['asc_tau_array_0', 'asc_tau_array_1']:
            inputs = input_features
            targets = np.argmin(np.abs(targets[:, None] - np.array([0.01, 0.1 / 3, 0.1, 1.0 / 3])[None, :]), axis=1)
            scaler = None
            gscv = GridSearchCV(classification_model(), classification_params)
        else:
            print(f'skipping col {col} due to unsupported dtype {targets.dtype}')
            continue
        perm = np.random.permutation(inputs.shape[0])
        gscv.fit(inputs[perm], targets[perm])
        score_dict[col] = gscv.best_score_
        score_std_dict[col] = gscv.cv_results_['std_test_score'][gscv.best_index_]
        # escore_dict[col] = gscv.score(inputs, targets)
        pred = gscv.predict(full_input_features)
        if scaler is None:
            vals = [0.01, 0.1 / 3, 0.1, 1.0 / 3]
            if pred.dtype != int:
                pred = pred.astype(int)
            pred = np.vectorize(lambda x: vals[x])(pred)
        else:
            pred = scaler.inverse_transform(pred)
        if col == 'spike_cut_length':
            pred = np.round(pred).astype(int)
        pred_truth_dict[col] = pred.tolist()
        if return_models:
            model_dict[col] = gscv
    if return_models:
        return score_dict, score_std_dict, pred_truth_dict, model_dict
    return score_dict, score_std_dict, pred_truth_dict

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

# %% Latents + cell types + morph

print('all three')
print(time.time())

# inputs = np.concatenate([latent_features, cell_features, morph_features], axis=1)
inputs = latent_features

targets = model_df[['El_reference', 'C', 'init_threshold', 'spike_cut_length', 'R_input']].to_numpy()
targets[:, 2] = targets[:, 2] * model_df['coeffs_th_inf'].to_numpy()
targets[:, 1] = np.log(targets[:, 1])
targets[:, 4] = np.log(targets[:, 4])
scaler = StandardScaler()
targets = scaler.fit_transform(targets)

# val_idx = np.random.choice(has_model.sum(), int(round(has_model.sum() / 5)), replace=False)
# val_mask = np.isin(np.arange(has_model.sum()), val_idx)
# train_mask = ~val_mask

np.random.seed(0)
n_folds = 5
if n_folds == 1:
    val_idx = np.random.choice(has_model.sum(), int(round(has_model.sum() / 5)), replace=False)
    val_mask = np.isin(np.arange(has_model.sum()), val_idx)
    train_masks = [~val_mask]
    val_masks = [val_mask]
else:
    idx = np.random.permutation(has_model.sum())
    samp_per_fold = has_model.sum() / n_folds
    train_masks = []
    val_masks = []
    for i in range(n_folds):
        val_idx = idx[round(i*samp_per_fold):round((i+1)*samp_per_fold)]
        val_mask = np.isin(np.arange(has_model.sum()), val_idx)
        train_mask = ~val_mask
        train_masks.append(train_mask)
        val_masks.append(val_mask)

# %%

def dict_mean_std(loss_dicts):
    mean_std_dict = {}
    for key in loss_dicts[0].keys():
        vals = [ld[key] for ld in loss_dicts]
        mean_std_dict[key + '_mean'] = np.mean(vals)
        mean_std_dict[key + '_std'] = np.std(vals)
    return mean_std_dict

def train_model(hidden_sizes=[32], lr=1e-4, nonlinearity='relu', dropout=0.1, weight_decay=1e-4, max_iter=2000, use_lr_scheduler=False, log_freq=200):
    torch.manual_seed(0)
    # np.random.seed(0)
    model = MLP(inputs.shape[1], layer_sizes=hidden_sizes, output_size=5, nonlinearity=nonlinearity, output_activation='none', dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    model_inputs = torch.from_numpy(inputs[has_model]).to(torch.float)
    model_targets = torch.from_numpy(targets).to(torch.float)

    loss_dicts = []
    models = []
    for train_mask, val_mask in zip(train_masks, val_masks):
        train_inputs = model_inputs[train_mask]
        val_inputs = model_inputs[val_mask]

        train_targets = model_targets[train_mask]
        val_targets = model_targets[val_mask]

        for it in range(max_iter):
            optimizer.zero_grad()
            train_pred = model(train_inputs)
            loss = F.mse_loss(train_pred, train_targets)
            loss.backward()
            optimizer.step()
            if use_lr_scheduler:
                lr_scheduler.step(loss)

            if (it + 1) % log_freq == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_inputs)
                    val_loss = F.mse_loss(val_pred, val_targets)
                val_r2 = r2_score(val_targets.detach().cpu().numpy(), val_pred.detach().cpu().numpy(), multioutput='raw_values')
                train_r2 = r2_score(train_targets.detach().cpu().numpy(), train_pred.detach().cpu().numpy(), multioutput='raw_values')
                total_r2 = r2_score(
                    np.concatenate([train_targets.detach().cpu().numpy(), val_targets.detach().cpu().numpy()], axis=0), 
                    np.concatenate([train_pred.detach().cpu().numpy(), val_pred.detach().cpu().numpy()], axis=0),
                    multioutput='raw_values'
                )
                print(f'Iter {it:03d}: train loss = {loss.item():.3f}, val loss = {val_loss.item():.3f}, ' + 
                    f'\ttotal R2 = {np.round(total_r2, 3)},\n\ttrain R2 = {np.round(train_r2, 3)}, val R2 = {np.round(val_r2, 3)}')
                model.train()
        model.eval()
        with torch.no_grad():
            val_pred = model(val_inputs)
            val_loss = F.mse_loss(val_pred, val_targets)
            pred = model(model_inputs)
            loss = F.mse_loss(pred, model_targets)
        val_r2 = r2_score(val_targets.detach().cpu().numpy(), val_pred.detach().cpu().numpy(), multioutput='raw_values')
        train_r2 = r2_score(train_targets.detach().cpu().numpy(), train_pred.detach().cpu().numpy(), multioutput='raw_values')
        model.train()
        loss_dict = {'train_loss': loss.item(), 'val_loss': val_loss.item(), 'total_loss': loss.item(),
            'train_r2_0': train_r2[0], 'train_r2_1': train_r2[1], 'train_r2_2': train_r2[2], 'train_r2_3': train_r2[3], 'train_r2_4': train_r2[4],
            'val_r2_0': val_r2[0], 'val_r2_1': val_r2[1], 'val_r2_2': val_r2[2], 'val_r2_3': val_r2[3], 'val_r2_4': val_r2[4],}
        loss_dicts.append(loss_dict)
        models.append(model)

    loss_dict = dict_mean_std(loss_dicts)

    best_idx = np.argmin([ld['total_loss'] for ld in loss_dicts])
    loss_dict['best_idx'] = best_idx
    # model = models[best_idx]

    return models, loss_dict

# _ = train_model()

# %%

search_space = {
    'dropout': lambda: np.random.random() * 0.6,
    'weight_decay': lambda: 10 ** (np.random.random() * -5 - 1),
    'lr': lambda: 10 ** (np.random.random() * -1.5 - 3),
    'hidden_size': lambda: np.random.choice([8, 16, 32, 64]),
    'hidden_depth': lambda: np.random.choice([1, 1, 1, 1, 2, 2, 3]),
    'nonlinearity': lambda: np.random.choice(['relu', 'relu', 'sigmoid']),
}

def sample_space(search_space):
    sample = {}
    for key, val in search_space.items():
        sample[key] = val()
    return sample

num_samples = 60

results = []
models = []
for idx in range(num_samples):
    print(f'sample {idx}')
    params = sample_space(search_space)
    hidden_sizes = [params['hidden_size']] * params['hidden_depth']
    model, losses = train_model(hidden_sizes, lr=params['lr'], dropout=params['dropout'], 
        weight_decay=params['weight_decay'], nonlinearity=params['nonlinearity'], max_iter=2000, log_freq=10000)
    params['run_id'] = idx
    params.update(losses)
    results.append(params)
    models.append(model)

results = pd.DataFrame(results)
# results.to_csv('../analysis/glif_scores/mlp.csv')

import pdb; pdb.set_trace()

# best_idx = results.val_loss_mean.argmin()

# best_models = models[best_idx]
# full_inputs = torch.from_numpy(inputs).to(torch.float)
# outputs = [model(full_inputs) for model in best_models]
# outputs = [o.detach().cpu().numpy() for o in outputs]
# def remap(output):
#     output = scaler.inverse_transform(output)
#     output[:, 1] = np.exp(output[:, 1])
#     output[:, 4] = np.exp(output[:, 4])
#     output[:, 3] = np.round(output[:, 3])
#     return output
# routputs = [remap(o) for o in outputs]

# for i, gp in enumerate(routputs):
#     df = pd.DataFrame(gp, index=dset.cell_ids, columns=['El_reference', 'C', 'init_threshold', 'spike_cut_length', 'R_input'])
#     df.reset_index(inplace=True)
#     df.to_csv(f'../analysis/glif_models/mlp{i}.csv')

exit(0)

# %%

print(time.time())
# import pdb; pdb.set_trace()

linear_scores = [llscore_dict, clscore_dict, mlscore_dict, lclscore_dict, lmlscore_dict, cmlscore_dict, lcmlscore_dict]
linear_scores = pd.DataFrame(linear_scores)
# linear_scores.to_csv('../analysis/glif_scores/linear_scores.csv')

linear_score_stds = [llscore_std_dict, clscore_std_dict, mlscore_std_dict, lclscore_std_dict, lmlscore_std_dict, cmlscore_std_dict, lcmlscore_std_dict]
linear_score_stds = pd.DataFrame(linear_score_stds)
# linear_score_stds.to_csv('../analysis/glif_scores/linear_score_stds.csv')

nonlinear_scores = [lnscore_dict, cnscore_dict, mnscore_dict, lcnscore_dict, lmnscore_dict, cmnscore_dict, lcmnscore_dict]
nonlinear_scores = pd.DataFrame(nonlinear_scores)
# nonlinear_scores.to_csv('../analysis/glif_scores/nonlinear_scores.csv')

nonlinear_score_stds = [lnscore_std_dict, cnscore_std_dict, mnscore_std_dict, lcnscore_std_dict, lmnscore_std_dict, cmnscore_std_dict, lcmnscore_std_dict]
nonlinear_score_stds = pd.DataFrame(nonlinear_score_stds)
# nonlinear_score_stds.to_csv('../analysis/glif_scores/nonlinear_score_stds.csv')

# %%



# %%

# linear_scores = pd.read_csv('../analysis/linear_scores.csv')
# linear_score_stds = pd.read_csv('../analysis/linear_score_stds.csv')
# nonlinear_scores = pd.read_csv('../analysis/nonlinear_scores.csv')
# nonlinear_score_stds = pd.read_csv('../analysis/nonlinear_score_stds.csv')

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
    plt.savefig(f'../analysis/glif_plots/{feature}.png')
    plt.close()

