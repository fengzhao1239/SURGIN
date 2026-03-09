import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import os



def load_hdf5(file_path):
    print(f">>>>> Loading data from {file_path}")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            val = f[key][:2000]
            data.update({key: val})
            print(f">>>>> Loaded {key} with shape {val.shape}")
    return data

class OperatorDataset(Dataset):
    def __init__(self, data, ps_flag):
        perm = data['permeability_log']
        ts = data['time_step']
        sat = data['saturation']
        pre = data['pressure']
        
        normed_perm = self._min_max_normalize(perm, 'perm_vertical_min_max.json')
        normed_ts = self._min_max_normalize(ts, 'ts_vertical_min_max.json')
        normed_sat = self._min_max_normalize(sat, 'sat_vertical_min_max.json')
        normed_pre = self._min_max_normalize(pre, 'pre_vertical_min_max.json')
        
        num_ts = normed_ts.shape[-1]
        num_x = normed_sat.shape[-2]
        num_y = normed_sat.shape[-1]
        perm_tiled = np.tile(normed_perm[:, None, :, :], (1, num_ts, 1, 1))
        ts_tiled = np.tile(normed_ts[:, :, None, None], (1, 1, num_x, num_y))
        assert perm_tiled.shape == ts_tiled.shape == normed_sat.shape == normed_pre.shape,\
            f"Shapes mismatch: {perm_tiled.shape}, {ts_tiled.shape}, {normed_sat.shape}, {normed_pre.shape}"
        
        x = np.stack((perm_tiled, ts_tiled), axis=-1)
        y = np.stack((normed_sat, normed_pre), axis=-1)
        
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x = self.x.permute(0, 2, 3, 1, 4)
        self.y = self.y.permute(0, 2, 3, 1, 4)
        self.num_samples = self.x.shape[0]
        if ps_flag == 'sat':
            self.y = self.y[..., 0]
        elif ps_flag == 'pre':
            self.y = self.y[..., 1]
        else:
            raise ValueError(f"Invalid ps_flag: {ps_flag}. Expected 'sat' or 'pre'.")
        print(f">>>>> Loaded dataset, x shape: {self.x.shape}, y shape: {self.y.shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def _min_max_normalize(self, data, save_name):
        if os.path.exists(f'dataset/{save_name}'):
            print(f">>>>> Loading min-max values from {save_name}")
            with open(f'dataset/{save_name}', 'r') as f:
                save_min_max = json.load(f)
            min_val = save_min_max['min']
            max_val = save_min_max['max']
        else:
            print(f">>>>> Calculating min-max values for {save_name}")
            min_val = np.min(data)
            max_val = np.max(data)
            save_min_max = {
                'min': float(min_val),
                'max': float(max_val)
            }
            with open(f'dataset/{save_name}', 'w') as f:
                json.dump(save_min_max, f, indent=4)
        return 2 * (data - min_val) / (max_val - min_val) - 1


if __name__ == '__main__':
    data = load_hdf5('dataset/Multi_Cartesian.hdf5')
    
    dataset = OperatorDataset(data)
    print(f">>>>> Dataset length: {len(dataset)}")
    print(f">>>>> First sample x shape: {dataset[0][0].shape}")
    print(f">>>>> First sample y shape: {dataset[0][1].shape}")
        