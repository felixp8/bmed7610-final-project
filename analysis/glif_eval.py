# %%

import os
import sys
os.chdir('../ssl_neuron/')

if len(sys.argv) > 2:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
else:
    start = 0
    end = 430

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import gc
import pickle
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import scipy.signal as signal
from allensdk.core.cell_types_cache import CellTypesCache

from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as json_utilities
from allensdk.model.glif.glif_neuron import GlifNeuron

from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor

from ssl_neuron.datasets import AllenDataset

import time
import uuid

# %%

config = json.load(open('./ssl_neuron/configs/config.json'))
config['data']['n_nodes'] = 1000

ctc = CellTypesCache(manifest_file='./ssl_neuron/data/cell_types/manifest.json', cache=True)

with open(f'../analysis/cell_ids.pkl', 'rb') as f:
    cell_ids = pickle.load(f)

glif_api = GlifApi()

# %%

llpred_truth_dict = pd.read_csv('../analysis/glif_models/ll.csv')
# lnpred_truth_dict = pd.read_csv('../analysis/glif_models/ln.csv')
# clpred_truth_dict = pd.read_csv('../analysis/glif_models/cl.csv')
# cnpred_truth_dict = pd.read_csv('../analysis/glif_models/cn.csv')
# mlpred_truth_dict = pd.read_csv('../analysis/glif_models/ml.csv')
# mnpred_truth_dict = pd.read_csv('../analysis/glif_models/mn.csv')
# lclpred_truth_dict = pd.read_csv('../analysis/glif_models/lcl.csv')
# lcnpred_truth_dict = pd.read_csv('../analysis/glif_models/lcn.csv')
# lmlpred_truth_dict = pd.read_csv('../analysis/glif_models/lml.csv')
# lmnpred_truth_dict = pd.read_csv('../analysis/glif_models/lmn.csv')
# cmlpred_truth_dict = pd.read_csv('../analysis/glif_models/cml.csv')
# cmnpred_truth_dict = pd.read_csv('../analysis/glif_models/cmn.csv')
# lcmlpred_truth_dict = pd.read_csv('../analysis/glif_models/lcml.csv')
# lcmnpred_truth_dict = pd.read_csv('../analysis/glif_models/lcmn.csv')

mlp0pred_truth_dict = pd.read_csv('../analysis/glif_models/ll.csv')
mlp1pred_truth_dict = pd.read_csv('../analysis/glif_models/ln.csv')
mlp2pred_truth_dict = pd.read_csv('../analysis/glif_models/cl.csv')
mlp3pred_truth_dict = pd.read_csv('../analysis/glif_models/cn.csv')
mlp3pred_truth_dict = pd.read_csv('../analysis/glif_models/ml.csv')

# df_list = [llpred_truth_dict, lnpred_truth_dict, clpred_truth_dict, 
#             cnpred_truth_dict, mlpred_truth_dict, mnpred_truth_dict, 
#             lclpred_truth_dict, lcnpred_truth_dict, lmlpred_truth_dict, 
#             lmnpred_truth_dict, cmlpred_truth_dict, cmnpred_truth_dict, 
#             lcmlpred_truth_dict, lcmnpred_truth_dict]

# label_list = ['ll', 'ln', 'cl', 'cn', 'ml', 'mn', 'lcl', 'lcn', 'lml', 'lmn', 'cml', 'cmn', 'lcml', 'lcmn']

# df_list = [lcmnpred_truth_dict]

# label_list = ['lcmn']

mlp0pred_truth_dict = pd.read_csv('../analysis/glif_models/mlp0.csv')
mlp1pred_truth_dict = pd.read_csv('../analysis/glif_models/mlp1.csv')
mlp2pred_truth_dict = pd.read_csv('../analysis/glif_models/mlp2.csv')
mlp3pred_truth_dict = pd.read_csv('../analysis/glif_models/mlp3.csv')
mlp4pred_truth_dict = pd.read_csv('../analysis/glif_models/mlp4.csv')

df_list = [mlp0pred_truth_dict, mlp1pred_truth_dict, mlp2pred_truth_dict, mlp3pred_truth_dict, mlp4pred_truth_dict]

label_list = ['mlp0', 'mlp1', 'mlp2', 'mlp3', 'mlp4']

has_model = llpred_truth_dict['has_model'].to_numpy()

neuron_config_base = {
    'El_reference': np.nan,
    'C': np.nan,
    'asc_amp_array': [0., 0.],
    'init_threshold': np.nan,
    'threshold_reset_method': {'params': {}, 'name': 'inf'},
    'th_inf': np.nan,
    'spike_cut_length': -1,
    'init_AScurrents': [0.0, 0.0],
    'init_voltage': 0.0,
    'threshold_dynamics_method': {'params': {}, 'name': 'inf'},
    'voltage_reset_method': {'params': {}, 'name': 'zero'},
    'extrapolation_method_name': 'endpoints',
    'dt': 5e-05,
    'voltage_dynamics_method': {'params': {}, 'name': 'linear_forward_euler'},
    'El': 0.0,
    'asc_tau_array': [0.01, 0.01],
    'R_input': np.nan,
    'AScurrent_dynamics_method': {'params': {}, 'name': 'exp'},
    'AScurrent_reset_method': {'params': {'r': [1.0, 1.0]}, 'name': 'sum'},
    'dt_multiplier': 10,
    'th_adapt': None,
    'coeffs': {
        'a': 1,
        'C': 1,
        'b': 1,
        'G': 1,
        'th_inf': 1.0,
        'asc_amp_array': [1.0, 1.0]
    },
    'type': 'GLIF'
}

# %%

def explained_variance(psth1, psth2):
    var1 = np.var(psth1)
    var2 = np.var(psth2)
    diffvar = np.var(psth1 - psth2)
    return (var1 + var2 - diffvar) / (var1 + var2)

def glif_explained_variance_ratio(sweep_stpsth, glif_spikes, sweep_var, kernel):
    glif_psth = signal.convolve(glif_spikes, kernel, mode='same')
    glif_var = 0
    for i in range(sweep_stpsth.shape[0]):
        glif_var += explained_variance(glif_psth, sweep_stpsth[i])
    return glif_var / sweep_var

# %%

# import pdb; pdb.set_trace()

results = []
for idx, cell_id in enumerate(cell_ids):
    if idx < start or idx >= end:
        continue
    print(f'{idx}: {cell_id}')
    print(time.time())
    cell_results = {'cell_id': int(cell_id)}

    cell_has_model = has_model[idx]

    temp_file = uuid.uuid4().hex
    data_set = ctc.get_ephys_data(cell_id, file_name=f'{temp_file}.nwb')

    sweep_info = []
    for sweep_number in data_set.get_sweep_numbers():
        md = data_set.get_sweep_metadata(sweep_number)
        md['sweep_number'] = sweep_number
        sweep_info.append(md)
    sweep_info = pd.DataFrame(sweep_info)

    for stimulus_name in [b'Noise 1']: # ], b'Noise 2']:
        print(time.time())
        stim_label = str(stimulus_name, "utf-8").lower().replace(" ", "_")
        save_dir = f'../analysis/sweep_data/{stim_label}/'

        sweep_numbers = sweep_info[sweep_info.aibs_stimulus_name == stimulus_name].sweep_number.tolist()
        if len(sweep_numbers) == 0:
            print(f"no sweeps found for neuron {cell_id} with stimulus {stimulus_name}")
            continue

        sampling_rate = None
        sweep_spike_times = []
        for sweep_number in sweep_numbers:
            sweep_data = data_set.get_sweep(sweep_number)

            index_range = sweep_data["index_range"]
            i = sweep_data["stimulus"][0:index_range[1]+1].copy() # in A
            v = sweep_data["response"][0:index_range[1]+1].copy() # in V
            i *= 1e12 # to pA
            v *= 1e3 # to mV

            if sampling_rate is None:
                sampling_rate = sweep_data["sampling_rate"] # in Hz
            else:
                assert sampling_rate == sweep_data["sampling_rate"]
            t = np.arange(0, len(v)) * (1.0 / sampling_rate)

            sweep_ext = EphysSweepFeatureExtractor(t=t, v=v, i=i) #, start=0, end=2.02)
            sweep_ext.process_spikes()

            spike_times = sweep_ext.spike_feature("threshold_t")
            sweep_spike_times.append(spike_times)
        
        print(f'sampling rate = {sampling_rate}')
        # if np.abs(sampling_rate - 200000) < 1e-4:
        #     print('No need to rerun')
        #     continue
        kern_sd = int(round(0.01 * sampling_rate))
        kernel = signal.gaussian(kern_sd * 7, kern_sd, sym=True)
        kernel /= kernel.sum()

        sweep_spikes = []
        for spike_times in sweep_spike_times:
            spike_idxs = np.round(spike_times * sweep_data["sampling_rate"]).astype(int)
            spikes = np.zeros(len(t))
            if np.any(spike_idxs > len(t)):
                assert 0, "spikes longer than stimulus"
            spikes[spike_idxs] = 1.
            sweep_spikes.append(spikes)
        
        sweep_stpsth = []
        for spikes in sweep_spikes:
            stpsth = signal.convolve(spikes, kernel, mode='same')
            sweep_stpsth.append(stpsth)
        sweep_psth = np.stack(sweep_stpsth).mean(axis=0)
        sweep_var = 0
        for stpsth in sweep_stpsth:
            sweep_var += explained_variance(sweep_psth, stpsth)
        
        sweep_stpsth = np.stack(sweep_stpsth)
        # np.savez(
        #     os.path.join(save_dir, f'{cell_id}.npz'),
        #     sweep_stpsth=sweep_stpsth,
        #     stimulus=sweep_data["stimulus"],
        #     sampling_rate=sweep_data["sampling_rate"],
        #     sweep_var=sweep_var,
        # )
        
        print(time.time())

        if cell_has_model:
            nm = glif_api.get_neuronal_models(cell_id)
            model_id = None
            var = None
            if len(nm) > 0:
                # print(f'{cell_id}*, ', end='')
                nm = nm[0]['neuronal_models']
                for model in nm:
                    if '3' in model['name'][:2]: # get basic LIF neurons
                        model_id = model['id']
                        try:
                            var = model['neuronal_model_runs'][0]['explained_variance_ratio']
                        except:
                            var = None
                        break
            
            if model_id is not None:

                neuron_config = glif_api.get_neuron_configs([model_id])[model_id]

                glif_neuron = GlifNeuron.from_dict(neuron_config)
                glif_neuron.dt = (1.0 / sampling_rate)

                stimulus = sweep_data["stimulus"]
                output = glif_neuron.run(stimulus)
                # spike_times = output['interpolated_spike_times']
                grid_spike_indices = output['spike_time_steps']

                t = np.arange(0, len(stimulus)) * glif_neuron.dt
                glif_spikes = np.zeros(len(t))
                if len(grid_spike_indices) > 0:
                    if grid_spike_indices.dtype != int:
                        grid_spike_indices = grid_spike_indices.astype(int)
                    glif_spikes[grid_spike_indices] = 1.

                var = glif_explained_variance_ratio(sweep_stpsth, glif_spikes, sweep_var, kernel)
        
                cell_results[f'{stim_label}_base_var'] = float(var)
        
        print(time.time())
            
        for model_label, model_df in zip(label_list, df_list):
            params = model_df.iloc[idx]
            neuron_config = copy.deepcopy(neuron_config_base)
            neuron_config['El_reference'] = params['El_reference']
            neuron_config['C'] = params['C']
            # neuron_config['asc_amp_array'] = [params['asc_amp_array_0'], params['asc_amp_array_1']]
            neuron_config['init_threshold'] = params['init_threshold']
            neuron_config['th_inf'] = params['init_threshold']
            neuron_config['spike_cut_length'] = params['spike_cut_length']
            # neuron_config['asc_tau_array'] = [params['asc_tau_array_0'], params['asc_tau_array_1']]
            neuron_config['R_input'] = params['R_input']
            # neuron_config['coeffs']['th_inf'] = params['coeffs_th_inf']
            
            # import pdb; pdb.set_trace()

            glif_neuron = GlifNeuron.from_dict(neuron_config)
            glif_neuron.dt = (1.0 / sampling_rate)

            stimulus = sweep_data["stimulus"]
            output = glif_neuron.run(stimulus)
            # spike_times = output['interpolated_spike_times']
            grid_spike_indices = output['spike_time_steps']

            t = np.arange(0, len(stimulus)) / sampling_rate # * glif_neuron.dt
            glif_spikes = np.zeros(len(t))
            if len(grid_spike_indices) > 0:
                if grid_spike_indices.dtype != int:
                    grid_spike_indices = grid_spike_indices.astype(int)
                glif_spikes[grid_spike_indices] = 1.
            # except:
            #     import pdb; pdb.set_trace()

            var = glif_explained_variance_ratio(sweep_stpsth, glif_spikes, sweep_var, kernel)
    
            cell_results[f'{stim_label}_{model_label}_var'] = float(var)

            gc.collect()
            
            print(time.time())

        gc.collect()
    
    # results.append(cell_results)
    if len(cell_results) > 1:
        with open(f'../analysis/glif_models/mlp_scores/{cell_id}.json', 'w') as f:
            json.dump(cell_results, f)
    
    os.remove(f'{temp_file}.nwb')
    gc.collect()

    # import pdb; pdb.set_trace()

print('done')

# %%

# results = pd.DataFrame(results)
# results.to_csv('../analysis/glif_models/scores.csv')

# import pdb; pdb.set_trace()
