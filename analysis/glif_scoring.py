# %%

import os
os.chdir('../ssl_neuron/')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import pickle
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

# %%

config = json.load(open('./ssl_neuron/configs/config.json'))
config['data']['n_nodes'] = 1000

ctc = CellTypesCache(manifest_file='./ssl_neuron/data/cell_types/manifest.json')

dset = AllenDataset(config, mode='all')

cell_idx = 0
cell_id = dset.cell_ids[cell_idx]

# %%

data_set = ctc.get_ephys_data(cell_id)

sweep_info = []
for sweep_number in data_set.get_sweep_numbers():
    md = data_set.get_sweep_metadata(sweep_number)
    md['sweep_number'] = sweep_number
    sweep_info.append(md)
sweep_info = pd.DataFrame(sweep_info)

stimulus_name = b'Noise 2'

noise1_sweep_numbers = sweep_info[sweep_info.aibs_stimulus_name == stimulus_name].sweep_number.tolist()
print(sweep_info[sweep_info.aibs_stimulus_name == stimulus_name].aibs_stimulus_amplitude_pa)

# %%

sampling_rate = None
sweep_spike_times = []
for sweep_number in noise1_sweep_numbers:
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

# %%

glif_api = GlifApi()

nm = glif_api.get_neuronal_models(cell_id)
if len(nm) < 1:
    # print(f'{cell_id}*, ', end='')
    assert 0, "No neuron models found"
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
    assert 0, "No neuron models found"

# %%

neuron_config = glif_api.get_neuron_configs([model_id])[model_id]

glif_neuron = GlifNeuron.from_dict(neuron_config)
glif_neuron.dt = (1.0 / sampling_rate)

stimulus = sweep_data["stimulus"][0:index_range[1]+1]
import time; print(time.time())
output = glif_neuron.run(stimulus)
print(time.time())
# spike_times = output['interpolated_spike_times']
grid_spike_indices = output['spike_time_steps']

# %%

t = np.arange(0, len(stimulus)) * glif_neuron.dt

glif_spikes = np.zeros(len(t))
glif_spikes[grid_spike_indices] = 1.

# %%

sweep_spikes = []
for spike_times in sweep_spike_times:
    spike_idxs = np.round(spike_times / glif_neuron.dt).astype(int)
    spikes = np.zeros(len(t))
    if np.any(spike_idxs > len(t)):
        assert 0, "spikes longer than stimulus"
    spikes[spike_idxs] = 1.
    sweep_spikes.append(spikes)

# %%

def explained_variance(psth1, psth2):
    var1 = np.var(psth1)
    var2 = np.var(psth2)
    diffvar = np.var(psth1 - psth2)
    return (var1 + var2 - diffvar) / (var1 + var2)

def explained_variance_ratio(sweep_spikes, glif_spikes, kern_sd_samp, kern_width_samp):
    kernel = signal.gaussian(kern_width_samp, kern_sd_samp, sym=True)
    kernel /= kernel.sum()
    glif_psth = signal.convolve(glif_spikes, kernel, mode='same')
    sweep_stpsth = []
    for spikes in sweep_spikes:
        stpsth = signal.convolve(spikes, kernel, mode='same')
        sweep_stpsth.append(stpsth)
    sweep_psth = np.stack(sweep_stpsth).mean(axis=0)
    glif_var = 0
    sweep_var = 0
    for stpsth in sweep_stpsth:
        glif_var += explained_variance(glif_psth, stpsth)
        sweep_var += explained_variance(sweep_psth, stpsth)
    return glif_var / sweep_var

# %%

print(f'Truth: {var:.6f}')
# for kern_sd_samp in [200, 600, 1000, 2000, 4000]:
#     ev = explained_variance_ratio(sweep_spikes, glif_spikes, kern_sd_samp, kern_sd_samp * 6)
#     print(f'{kern_sd_samp}: {ev:.6f}')
kern_sd_samp = 2000 # 10 ms, best match
ev = explained_variance_ratio(sweep_spikes, glif_spikes, kern_sd_samp, kern_sd_samp * 6)
print(f'{kern_sd_samp}: {ev:.6f}')

# import pdb; pdb.set_trace()
