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
latent_df = pd.DataFrame(latent_features, index=dset.cell_ids)

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

# %%

def fit_eval_decoder(input_features, target_df, skip_cols=[], to_str_cols=[], 
                     regression_model=Ridge, regression_params={'alpha': np.logspace(-8, 3, 12)},
                     classification_model=LogisticRegression, classification_params={},
                     seed=0, return_models=False):
    np.random.seed(seed)
    score_dict = {}
    score_std_dict = {}
    pred_truth_dict = {}
    model_dict = {}
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
        if return_models:
            model_dict[col] = gscv
    if return_models:
        return score_dict, score_std_dict, pred_truth_dict, model_dict
    else:
        return score_dict, score_std_dict, pred_truth_dict

# %%

print('latents -> cell')
print(time.time())

inputs = latent_features

include_cols = ['reporter_status',
       'structure_layer_name', 'structure_area_id', 'structure_area_abbrev',
       'transgenic_line', 
       'dendrite_type', 'apical', 'reconstruction_type',
       'structure_hemisphere',
       'normalized_depth']
skip_cols = [col for col in cell_df.columns if col not in include_cols]

lclscore_dict, lclscore_std_dict, lclpred_truth_dict = fit_eval_decoder(
    inputs, cell_df, skip_cols=skip_cols, to_str_cols=['structure_area_id'],
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0,
)

lcnscore_dict, lcnscore_std_dict, lcnpred_truth_dict = fit_eval_decoder(
    inputs, cell_df, skip_cols=skip_cols, to_str_cols=['structure_area_id'],
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0,
)

# %%

print('morph -> cell')
print(time.time())

inputs = morph_features

include_cols = ['reporter_status',
       'structure_layer_name', 'structure_area_id', 'structure_area_abbrev',
       'transgenic_line', 
       'dendrite_type', 'apical', 'reconstruction_type',
       'structure_hemisphere',
       'normalized_depth']
skip_cols = [col for col in cell_df.columns if col not in include_cols]

mclscore_dict, mclscore_std_dict, mclpred_truth_dict = fit_eval_decoder(
    inputs, cell_df, skip_cols=skip_cols, to_str_cols=['structure_area_id'],
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0,
)

mcnscore_dict, mcnscore_std_dict, mcnpred_truth_dict = fit_eval_decoder(
    inputs, cell_df, skip_cols=skip_cols, to_str_cols=['structure_area_id'],
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0,
)

# %%

print('latents -> morph')
print(time.time())

inputs = latent_features

include_cols = ['average_bifurcation_angle_local',
       'average_contraction', 'average_diameter', 'average_fragmentation',
       'average_parent_daughter_ratio',
       'max_branch_order', 'max_euclidean_distance', 'max_path_distance',
       'number_bifurcations', 'number_branches',
       'number_nodes', 'number_stems', 'number_tips', 'overall_depth',
       'overall_height', 'overall_width', 'soma_surface', 'total_length',
       'total_surface', 'total_volume']
skip_cols = [col for col in morph_df.columns if col not in include_cols]

lmlscore_dict, lmlscore_std_dict, lmlpred_truth_dict = fit_eval_decoder(
    inputs, morph_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0
)

lmnscore_dict, lmnscore_std_dict, lmnpred_truth_dict = fit_eval_decoder(
    inputs, morph_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

# %%

print('cell -> morph')
print(time.time())

inputs = cell_features

include_cols = ['average_bifurcation_angle_local',
       'average_contraction', 'average_diameter', 'average_fragmentation',
       'average_parent_daughter_ratio',
       'max_branch_order', 'max_euclidean_distance', 'max_path_distance',
       'number_bifurcations', 'number_branches',
       'number_nodes', 'number_stems', 'number_tips', 'overall_depth',
       'overall_height', 'overall_width', 'soma_surface', 'total_length',
       'total_surface', 'total_volume']
skip_cols = [col for col in morph_df.columns if col not in include_cols]

cmlscore_dict, cmlscore_std_dict, cmlpred_truth_dict = fit_eval_decoder(
    inputs, morph_df, skip_cols=skip_cols, 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0
)

cmnscore_dict, cmnscore_std_dict, cmnpred_truth_dict = fit_eval_decoder(
    inputs, morph_df, skip_cols=skip_cols, 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

# %%

print('cell -> latent')
print(time.time())

inputs = cell_features

cllscore_dict, cllscore_std_dict, cllpred_truth_dict = fit_eval_decoder(
    inputs, latent_df, skip_cols=[], 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0
)

clnscore_dict, clnscore_std_dict, clnpred_truth_dict = fit_eval_decoder(
    inputs, latent_df, skip_cols=[], 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

# %%

print('morph -> latent')
print(time.time())

inputs = morph_features

mllscore_dict, mllscore_std_dict, mllpred_truth_dict = fit_eval_decoder(
    inputs, latent_df, skip_cols=[], 
    regression_model=Ridge, regression_params={'alpha': np.logspace(-6, 4, 22)},
    classification_model=LogisticRegression, classification_params={},
    seed=0
)

mlnscore_dict, mlnscore_std_dict, mlnpred_truth_dict = fit_eval_decoder(
    inputs, latent_df, skip_cols=[], 
    regression_model=SVR, regression_params={'C': np.logspace(-7, 3, 22)},
    classification_model=SVC, classification_params={'C': np.logspace(-7, 3, 22)},
    seed=0
)

# %%

latent_scores = []
latent_err = []
latent_source = []
morph_scores = []
morph_err = []
morph_source = []
features = []
for feat in lclscore_dict.keys():
    features.append(feat)

    latent_linear_score = lclscore_dict[feat]
    latent_nonlinear_score = lcnscore_dict[feat]

    if latent_linear_score >= latent_nonlinear_score:
        latent_scores.append(latent_linear_score)
        latent_err.append(lclscore_std_dict[feat])
        latent_source.append('l')
    else:
        latent_scores.append(latent_nonlinear_score)
        latent_err.append(lcnscore_std_dict[feat])
        latent_source.append('n')

    morph_linear_score = mclscore_dict[feat]
    morph_nonlinear_score = mcnscore_dict[feat]

    if morph_linear_score >= morph_nonlinear_score:
        morph_scores.append(morph_linear_score)
        morph_err.append(mclscore_std_dict[feat])
        morph_source.append('l')
    else:
        morph_scores.append(morph_nonlinear_score)
        morph_err.append(mcnscore_std_dict[feat])
        morph_source.append('n')

# %%

plt.bar(np.arange(len(features)) - 0.2, latent_scores, 0.4, yerr=np.array(latent_err) / np.sqrt(5), label='morph_latent')
plt.bar(np.arange(len(features)) + 0.2, morph_scores, 0.4, yerr=np.array(morph_err) / np.sqrt(5), label='morph_feature')
plt.xticks(ticks=np.arange(len(features)), labels=features, rotation=90)
plt.ylabel('R^2 or classification accuracy')
plt.legend()
plt.title('Cell Type Feature Prediction')
plt.tight_layout()
plt.savefig('../analysis/cell_pred.png')
plt.clf()

print(latent_source)
print(morph_source)

# %%

latent_scores = []
latent_err = []
latent_source = []
cell_scores = []
cell_err = []
cell_source = []
features = []
for feat in lmlscore_dict.keys():
    features.append(feat)

    latent_linear_score = lmlscore_dict[feat]
    latent_nonlinear_score = lmnscore_dict[feat]

    if latent_linear_score >= latent_nonlinear_score:
        latent_scores.append(latent_linear_score)
        latent_err.append(lmlscore_std_dict[feat])
        latent_source.append('l')
    else:
        latent_scores.append(latent_nonlinear_score)
        latent_err.append(lmnscore_std_dict[feat])
        latent_source.append('n')

    cell_linear_score = cmlscore_dict[feat]
    cell_nonlinear_score = cmnscore_dict[feat]

    if cell_linear_score >= cell_nonlinear_score:
        cell_scores.append(cell_linear_score)
        cell_err.append(cmlscore_std_dict[feat])
        cell_source.append('l')
    else:
        cell_scores.append(cell_nonlinear_score)
        cell_err.append(cmnscore_std_dict[feat])
        cell_source.append('n')

# %%

plt.bar(np.arange(len(features)) - 0.2, latent_scores, 0.4, yerr=np.array(latent_err) / np.sqrt(5), label='morph_latent')
plt.bar(np.arange(len(features)) + 0.2, cell_scores, 0.4, yerr=np.array(cell_err) / np.sqrt(5), label='cell_type')
plt.xticks(ticks=np.arange(len(features)), labels=features, rotation=90)
plt.ylabel('R^2 or classification accuracy')
plt.legend()
plt.title('Morphology Feature Prediction')
plt.tight_layout()
plt.savefig('../analysis/morph_pred.png')
plt.clf()

print(latent_source)
print(cell_source)

# %%

morph_scores = []
morph_err = []
morph_source = []
cell_scores = []
cell_err = []
cell_source = []
features = []
for feat in cllscore_dict.keys():
    features.append(feat)

    cell_linear_score = cllscore_dict[feat]
    cell_nonlinear_score = clnscore_dict[feat]

    if cell_linear_score >= cell_nonlinear_score:
        cell_scores.append(cell_linear_score)
        cell_err.append(cllscore_std_dict[feat])
        cell_source.append('l')
    else:
        cell_scores.append(cell_nonlinear_score)
        cell_err.append(clnscore_std_dict[feat])
        cell_source.append('n')

    morph_linear_score = mllscore_dict[feat]
    morph_nonlinear_score = mlnscore_dict[feat]

    if morph_linear_score >= morph_nonlinear_score:
        morph_scores.append(morph_linear_score)
        morph_err.append(mllscore_std_dict[feat])
        morph_source.append('l')
    else:
        morph_scores.append(morph_nonlinear_score)
        morph_err.append(mlnscore_std_dict[feat])
        morph_source.append('n')

# %%

plt.bar(np.arange(len(features)) - 0.2, cell_scores, 0.4, yerr=np.array(cell_err) / np.sqrt(5), label='cell_type')
plt.bar(np.arange(len(features)) + 0.2, morph_scores, 0.4, yerr=np.array(morph_err) / np.sqrt(5), label='morph_feature')
plt.xticks(ticks=np.arange(len(features)))
plt.ylabel('R^2 or classification accuracy')
plt.legend()
plt.title('Morphology Latent Prediction')
plt.tight_layout()
plt.savefig('../analysis/latent_pred.png')
plt.clf()

print(cell_source)
print(morph_source)