'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import json
import numpy as np
from einops import rearrange
import h5py
from scipy.stats import qmc
import warnings

from Surrogate.ufno import Net3d

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass
    

# =====================================
# Operators for 2D Cartesian data
# =====================================

def load_test_hdf5(file_path):
    print(f">>>>> Loading test data from {file_path}")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key in ["pressure", "saturation", "permeability_log"]:
                val = f[key][5000:]    # ! testing dataset
                data.update({key: val})
                # print(f">>>>> Loaded test {key} with shape {val.shape}")
    return data

def load_test_hdf5_vertical(file_path):
    print(f">>>>> Loading test data from {file_path}")
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key in ["pressure", "saturation", "permeability_log"]:
                val = f[key][2000:]    # ! testing dataset
                data.update({key: val})
                # print(f">>>>> Loaded test {key} with shape {val.shape}")
    return data

def retrieve_min_max(name):
        with open(f'/ehome/zhao/DiffNO/dataset/{name}_min_max.json', 'r') as f:
            min_max = json.load(f)
            min = min_max['min']
            max = min_max['max']
        return min, max
        
def norm(data, min_val, max_val):
    return -1 + (data - min_val) * 2. / (max_val - min_val)

def unnorm(norm_data, min_val, max_val):
    return (norm_data + 1) * (max_val - min_val) / 2 + min_val


# =====================================
# Horizontal case
# =====================================
@register_operator(name='horizontal_superresolution')
class horizontal_superresolution(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                ds_size,
                condition_variable,
                vanilla_flag=False,
                ) -> None:
        assert condition_variable in ['sat', 'pre', 'perm'], "Condition variable must be either 'sat' or 'pre' or 'perm'."
        
        self.device = device
        self.vanilla_flag = vanilla_flag
        self.ds_size = (ds_size, ds_size) if isinstance(ds_size, int) else tuple(ds_size)    # todo: specify ur downsampling size
        
        test_sim = load_test_hdf5(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        if condition_variable == 'sat':
            self.surrogate = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
            self.surrogate.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat.pth')['model_state_dict'])
            self.surrogate.eval()
            self.flag = 'sat'
            self.foi = self.sat
        elif condition_variable == 'pre':
            self.surrogate = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
            self.surrogate.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre.pth')['model_state_dict'])
            self.surrogate.eval()
            self.flag = 'pre'
            self.foi = self.pre
        else:
            self.surrogate = None
            self.flag = 'perm'
            self.foi = self.perm
        
        tn = [1., 6., 12., 24., 36., 60., 7*12., 10*12., 15*12, 20*12]  # 10 time steps in months
        ts = [t*86400*30 for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts')[0], retrieve_min_max('ts')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, pre_norm=False, out_unnorm=False, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        if pre_norm:
            warnings.warn("UFNO pre_norm use warning: you are using prenorm for K, which should only be used when you are directly calling UFNO for evaluation, instead of diffusion sampling.")
            data = norm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate = self.surrogate(x_surrogate)    # <b, 64, 64, t=10>
        if len(out_surrogate.shape) < 4:
            out_surrogate = out_surrogate.unsqueeze(0)
        if out_unnorm:
            warnings.warn("UFNO out_unnorm use warning: you unnorming UFNO outputs, which should only be used when you are directly calling UFNO for evaluation, instead of diffusion sampling.")
            out_surrogate = unnorm(out_surrogate, retrieve_min_max(f'{self.flag}')[0], retrieve_min_max(f'{self.flag}')[1])
        return out_surrogate
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        if self.surrogate is not None:
            out = self.surrogate_forward(data)
            out = rearrange(out, 'b h w t -> b t h w')
        else:
            out = data

        if self.vanilla_flag:
            down = out
        else:
            down = F.interpolate(out, size=self.ds_size, mode='nearest')
        # print(f'forward output shape: {down.shape}')
        return down
        
    def measurement(self):
        normed_measurement = norm(self.foi, retrieve_min_max(f'{self.flag}')[0], retrieve_min_max(f'{self.flag}')[1])
        normed_measurement = torch.tensor(normed_measurement, dtype=torch.float32).to(self.device)[None, ...]
        if len(normed_measurement.shape) == 3:
            normed_measurement = normed_measurement.unsqueeze(0)
        assert len(normed_measurement.shape) == 4, f"Expected 4D tensor, got {normed_measurement.shape} tensor"
        if self.vanilla_flag:
            self.down = normed_measurement
        else:
            self.down = F.interpolate(normed_measurement, size=self.ds_size, mode='nearest')
        # print(f'Measurement shape: {self.down.shape}')
        return self.down
    
    def retrieve_reference(self):
        reference = {
            'condition': unnorm(self.down, retrieve_min_max(f'{self.flag}')[0], retrieve_min_max(f'{self.flag}')[1]).detach().cpu().squeeze().numpy(),
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving condition and reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference


@register_operator(name='horizontal_sparse_well')
class horizontal_sparse_well(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                well_num,
                with_perm=False,
                ) -> None:
        
        self.device = device
        if well_num == 1:
            self.query_idx = [[40, 40]]
        elif well_num == 4:
            self.query_idx = [[40, 40], [24, 24], [24, 40], [40, 24]]
        elif well_num == 8:
            self.query_idx = [[24, 24], [24, 32], [24, 40],
                              [32, 24], [32, 40],
                              [40, 24], [40, 32], [40, 40]]
        else:
            raise ValueError("Well number must be either 1, 4, or 8.")
        self.with_perm = with_perm
        
        test_sim = load_test_hdf5(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1., 6., 12., 24., 36., 60., 7*12., 10*12., 15*12, 20*12]  # 10 time steps in months
        ts = [t*86400*30 for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts')[0], retrieve_min_max('ts')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        if self.with_perm:
            whole = torch.stack([out_sg, out_p, perm], dim=1)    # <b, c=3, t, h, w>
        else:
            whole = torch.stack([out_sg, out_p], dim=1)    # <b, c=2, t, h, w>
        sparse = []
        for h_idx, w_idx in self.query_idx:
            sparse.append(whole[:, :, :, h_idx, w_idx])    # list of <b, c, t>
        sparse = torch.stack(sparse, dim=0).permute(1, 2, 3, 0)    # <b, c, t, well_num>
        return sparse
        
    def measurement(self):
        norm_sat = torch.tensor(norm(self.sat, retrieve_min_max('sat')[0], retrieve_min_max('sat')[1])).to(self.device)
        norm_pre = torch.tensor(norm(self.pre, retrieve_min_max('pre')[0], retrieve_min_max('pre')[1])).to(self.device)
        norm_perm = torch.tensor(norm(self.perm, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        if self.with_perm:
            whole = torch.stack([norm_sat, norm_pre, norm_perm], dim=0)    # <c=3, t, h, w>
        else:
            whole = torch.stack([norm_sat, norm_pre], dim=0)    # <c=2, t, h, w>
        sparse = []
        for h_idx, w_idx in self.query_idx:
            sparse.append(whole[:, :, h_idx, w_idx])    # list of <c, t>
        sparse = torch.stack(sparse, dim=0).permute(1, 2, 0)    # <c, t, well_num>
        sparse = sparse.unsqueeze(0)    # <1, c, t, well_num>
        return sparse
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference


# =====================================
# Vertical case
# =====================================
@register_operator(name='vertical_sparse_well')
class vertical_sparse_well(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                end_t_idx=4,
                with_perm=False,
                ) -> None:
        
        self.device = device
        self.with_perm = with_perm
        self.end_t_idx = end_t_idx
        
        test_sim = load_test_hdf5_vertical(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat_vertical.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre_vertical.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1*12, 2*12, 3*12, 4*12, 5*12, 6*12, 7*12, 8*12, 9*12, 10*12]  # 10 time steps in months
        ts = [t*86400*30. for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts_vertical')[0], retrieve_min_max('ts_vertical')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        if self.with_perm:
            whole = torch.stack([out_sg, out_p, perm], dim=1)    # <b, c=3, t, h, w>
        else:
            whole = torch.stack([out_sg, out_p], dim=1)    # <b, c=2, t, h, w>

        well_1 = whole[:, :, :self.end_t_idx, :, 24]
        well_2 = whole[:, :, :self.end_t_idx, :, 32]
        well_3 = whole[:, :, :self.end_t_idx, :, 40]    # <b, c, t, h>
        
        sparse = torch.cat([well_1, well_2, well_3], dim=-1)    # <b, c, t, 3*h>
        return sparse
        
    def measurement(self):
        norm_sat = torch.tensor(norm(self.sat, retrieve_min_max('sat_vertical')[0], retrieve_min_max('sat_vertical')[1])).to(self.device)
        norm_pre = torch.tensor(norm(self.pre, retrieve_min_max('pre_vertical')[0], retrieve_min_max('pre_vertical')[1])).to(self.device)
        norm_perm = torch.tensor(norm(self.perm, retrieve_min_max('perm_vertical')[0], retrieve_min_max('perm_vertical')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        if self.with_perm:
            whole = torch.stack([norm_sat, norm_pre, norm_perm], dim=0)    # <c=3, t, h, w>
        else:
            whole = torch.stack([norm_sat, norm_pre], dim=0)    # <c=2, t, h, w>
            
        well_1 = whole[:, :self.end_t_idx, :, 24]
        well_2 = whole[:, :self.end_t_idx, :, 32]
        well_3 = whole[:, :self.end_t_idx, :, 40]    # <c, t, h>
        
        sparse = torch.cat([well_1, well_2, well_3], dim=-1)    # <c, t, 3*h>
        assert sparse.size(-1) == 3 * whole.size(-1), f"Expected last dim size {3 * whole.size(-1)}, got {sparse.size(-1)}"
        sparse = sparse.unsqueeze(0)    # <1, c, t, 3*h>
        print(f'Measurement shape: {sparse.shape}')
        return sparse
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference


@register_operator(name='vertical_sparse_well_noise')
class vertical_sparse_well_noise(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                end_t_idx=4,
                with_perm=False,
                ) -> None:
        
        self.device = device
        self.with_perm = with_perm
        self.end_t_idx = end_t_idx
        
        test_sim = load_test_hdf5_vertical(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat_vertical.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre_vertical.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1*12, 2*12, 3*12, 4*12, 5*12, 6*12, 7*12, 8*12, 9*12, 10*12]  # 10 time steps in months
        ts = [t*86400*30. for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts_vertical')[0], retrieve_min_max('ts_vertical')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        if self.with_perm:
            whole = torch.stack([out_sg, out_p, perm], dim=1)    # <b, c=3, t, h, w>
        else:
            whole = torch.stack([out_sg, out_p], dim=1)    # <b, c=2, t, h, w>

        well_1 = whole[:, :, :self.end_t_idx, :, 24]
        well_2 = whole[:, :, :self.end_t_idx, :, 32]
        well_3 = whole[:, :, :self.end_t_idx, :, 40]    # <b, c, t, h>
        
        sparse = torch.cat([well_1, well_2, well_3], dim=-1)    # <b, c, t, 3*h>
        return sparse
        
    def measurement(self):
        sat = self.sat + np.random.randn(*self.sat.shape) * 0.1  # Adding noise to saturation
        pre = self.pre + np.random.randn(*self.pre.shape) * (19e6 * 0.1)  # Adding noise to pressure
        perm = self.perm + np.random.randn(*self.perm.shape) * (4.5 * 0.1)  # Adding noise to permeability
        
        norm_sat = torch.tensor(norm(sat, retrieve_min_max('sat_vertical')[0], retrieve_min_max('sat_vertical')[1])).to(self.device)
        norm_pre = torch.tensor(norm(pre, retrieve_min_max('pre_vertical')[0], retrieve_min_max('pre_vertical')[1])).to(self.device)
        norm_perm = torch.tensor(norm(perm, retrieve_min_max('perm_vertical')[0], retrieve_min_max('perm_vertical')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        if self.with_perm:
            whole = torch.stack([norm_sat, norm_pre, norm_perm], dim=0)    # <c=3, t, h, w>
        else:
            whole = torch.stack([norm_sat, norm_pre], dim=0)    # <c=2, t, h, w>
            
        well_1 = whole[:, :self.end_t_idx, :, 24]
        well_2 = whole[:, :self.end_t_idx, :, 32]
        well_3 = whole[:, :self.end_t_idx, :, 40]    # <c, t, h>
        
        sparse = torch.cat([well_1, well_2, well_3], dim=-1)    # <c, t, 3*h>
        assert sparse.size(-1) == 3 * whole.size(-1), f"Expected last dim size {3 * whole.size(-1)}, got {sparse.size(-1)}"
        sparse = sparse.unsqueeze(0)    # <1, c, t, 3*h>
        print(f'Measurement shape: {sparse.shape}')
        return sparse
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference


@register_operator(name='vertical_sparse_well_5')
class vertical_sparse_well_5(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                end_t_idx=4,
                with_perm=False,
                ) -> None:
        
        self.device = device
        self.with_perm = with_perm
        self.end_t_idx = end_t_idx
        
        test_sim = load_test_hdf5_vertical(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat_vertical.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre_vertical.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1*12, 2*12, 3*12, 4*12, 5*12, 6*12, 7*12, 8*12, 9*12, 10*12]  # 10 time steps in months
        ts = [t*86400*30. for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts_vertical')[0], retrieve_min_max('ts_vertical')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        if self.with_perm:
            whole = torch.stack([out_sg, out_p, perm], dim=1)    # <b, c=3, t, h, w>
        else:
            whole = torch.stack([out_sg, out_p], dim=1)    # <b, c=2, t, h, w>

        well_1 = whole[:, :, :self.end_t_idx, :, 24]
        well_2 = whole[:, :, :self.end_t_idx, :, 32]
        well_3 = whole[:, :, :self.end_t_idx, :, 40]    # <b, c, t, h>
        well_4 = whole[:, :, :self.end_t_idx, :, 48]
        well_5 = whole[:, :, :self.end_t_idx, :, 16]
        
        sparse = torch.cat([well_1, well_2, well_3, well_4, well_5], dim=-1)    # <b, c, t, 3*h>
        return sparse
        
    def measurement(self):
        norm_sat = torch.tensor(norm(self.sat, retrieve_min_max('sat_vertical')[0], retrieve_min_max('sat_vertical')[1])).to(self.device)
        norm_pre = torch.tensor(norm(self.pre, retrieve_min_max('pre_vertical')[0], retrieve_min_max('pre_vertical')[1])).to(self.device)
        norm_perm = torch.tensor(norm(self.perm, retrieve_min_max('perm_vertical')[0], retrieve_min_max('perm_vertical')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        if self.with_perm:
            whole = torch.stack([norm_sat, norm_pre, norm_perm], dim=0)    # <c=3, t, h, w>
        else:
            whole = torch.stack([norm_sat, norm_pre], dim=0)    # <c=2, t, h, w>
            
        well_1 = whole[:, :self.end_t_idx, :, 24]
        well_2 = whole[:, :self.end_t_idx, :, 32]
        well_3 = whole[:, :self.end_t_idx, :, 40]    # <c, t, h>
        well_4 = whole[:, :self.end_t_idx, :, 48]
        well_5 = whole[:, :self.end_t_idx, :, 16]
        
        sparse = torch.cat([well_1, well_2, well_3, well_4, well_5], dim=-1)    # <c, t, 3*h>
        assert sparse.size(-1) == 5 * whole.size(-1), f"Expected last dim size {5 * whole.size(-1)}, got {sparse.size(-1)}"
        sparse = sparse.unsqueeze(0)    # <1, c, t, 3*h>
        print(f'Measurement shape: {sparse.shape}')
        return sparse
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference


@register_operator(name='vertical_inpainting')
class vertical_inpainting(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                condition_variable
                ) -> None:
        
        self.device = device
        self.condition_variable = condition_variable
        
        test_sim = load_test_hdf5_vertical(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat_vertical.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre_vertical.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1*12, 2*12, 3*12, 4*12, 5*12, 6*12, 7*12, 8*12, 9*12, 10*12]  # 10 time steps in months
        ts = [t*86400*30. for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts_vertical')[0], retrieve_min_max('ts_vertical')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        
        mask = torch.ones_like(out_sg)
        mask[:, :, 16:48, 16:48] = 0.0  # Central 32x32 region
        
        if self.condition_variable == 'sat':
            out = out_sg
        elif self.condition_variable == 'pre':
            out = out_p
        elif self.condition_variable == 'perm':
            out = perm
        else:
            raise ValueError(f"Unknown condition variable: {self.condition_variable}")

        sparse = out * mask
        return sparse
        
    def measurement(self):
        norm_sat = torch.tensor(norm(self.sat, retrieve_min_max('sat_vertical')[0], retrieve_min_max('sat_vertical')[1])).to(self.device)
        norm_pre = torch.tensor(norm(self.pre, retrieve_min_max('pre_vertical')[0], retrieve_min_max('pre_vertical')[1])).to(self.device)
        norm_perm = torch.tensor(norm(self.perm, retrieve_min_max('perm_vertical')[0], retrieve_min_max('perm_vertical')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        
        mask = torch.ones_like(norm_sat)
        mask[:, 16:48, 16:48] = 0.0  # Central 32x32 region
        
        if self.condition_variable == 'sat':
            whole = norm_sat
        elif self.condition_variable == 'pre':
            whole = norm_pre
        elif self.condition_variable == 'perm':
            whole = norm_perm
        else:
            raise ValueError(f"Unknown condition variable: {self.condition_variable}")
        sparse = whole * mask
        sparse = sparse.unsqueeze(0)
        print(f'Measurement shape: {sparse.shape}')
        return sparse
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference




@register_operator(name='vertical_allknown')
class vertical_allknown(NonLinearOperator):
    def __init__(self,
                device,
                hdf5_path,
                test_idx,
                condition_space
                ) -> None:
        
        self.device = device
        self.condition_space = condition_space
        
        test_sim = load_test_hdf5_vertical(hdf5_path)
        self.perm = test_sim['permeability_log'][test_idx]
        self.sat = test_sim['saturation'][test_idx]
        self.pre = test_sim['pressure'][test_idx]
        print(f"Test data loaded, with shape K: {self.perm.shape}, Sg: {self.sat.shape}, P: {self.pre.shape}")
        
        self.surrogate_sg = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_sg.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_sat_vertical.pth')['model_state_dict'])
        self.surrogate_sg.eval()

        self.surrogate_p = Net3d(modes1=10, modes2=10, modes3=8, width=36).to(self.device)
        self.surrogate_p.load_state_dict(torch.load('/ehome/zhao/DiffNO/checkpoint/ufno_pre_vertical.pth')['model_state_dict'])
        self.surrogate_p.eval()
        
        tn = [1*12, 2*12, 3*12, 4*12, 5*12, 6*12, 7*12, 8*12, 9*12, 10*12]  # 10 time steps in months
        ts = [t*86400*30. for t in tn]
        normed_ts = norm(np.array(ts), retrieve_min_max('ts_vertical')[0], retrieve_min_max('ts_vertical')[1])
        normed_ts = np.tile(normed_ts[:, None, None], (1, self.sat.shape[-2], self.sat.shape[-1]))    # <1, 64, 64>
        self.normed_ts = torch.tensor(normed_ts, dtype=torch.float32).to(self.device)
    
    def surrogate_forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        batch_size = data.shape[0]
        ts_expand = self.normed_ts.expand(batch_size, -1, -1, -1)    # <b, t, 64, 64>
        data_expand = data.expand(-1, ts_expand.size(1), -1, -1)    # <b, c=t, 64, 64>
        assert ts_expand.shape == data_expand.shape, f"UFNO input shapes mismatch: {ts_expand.shape}, {data_expand.shape}"
        
        # * surrogate model
        # ! for UFNO, the input shape is <B, H, W, T, C>
        x_surrogate = torch.stack((data_expand, ts_expand), dim=-1)    # <b, t, 64, 64, c=2>
        x_surrogate = rearrange(x_surrogate, 'b t h w c -> b h w t c')
        out_surrogate_sg = self.surrogate_sg(x_surrogate)    # <b, 64, 64, t=10>
        out_surrogate_p = self.surrogate_p(x_surrogate)    # <b, 64, 64, t=10>
        
        if len(out_surrogate_sg.shape) < 4:
            out_surrogate_sg = out_surrogate_sg.unsqueeze(0)
        if len(out_surrogate_p.shape) < 4:
            out_surrogate_p = out_surrogate_p.unsqueeze(0)
        return out_surrogate_sg, out_surrogate_p
        
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, 64, 64>
        out_sg, out_p = self.surrogate_forward(data)
        out_sg = rearrange(out_sg, 'b h w t -> b t h w')
        out_p = rearrange(out_p, 'b h w t -> b t h w')
        # perm = unnorm(data, retrieve_min_max('perm')[0], retrieve_min_max('perm')[1])
        perm = data.expand(-1, out_sg.size(1), -1, -1)
        assert out_sg.shape == out_p.shape == perm.shape, f"Variables shapes mismatch: {out_sg.shape}, {out_p.shape}, {perm.shape}"
        
        if self.condition_space == 'solution':
            out = torch.stack([out_sg, out_p], dim=1)
            # out = out_sg
        elif self.condition_space == 'parameter':
            out = perm
        else:
            raise ValueError(f"Unknown condition space: {self.condition_space}")

        return out
        
    def measurement(self):
        norm_sat = torch.tensor(norm(self.sat, retrieve_min_max('sat_vertical')[0], retrieve_min_max('sat_vertical')[1])).to(self.device)
        norm_pre = torch.tensor(norm(self.pre, retrieve_min_max('pre_vertical')[0], retrieve_min_max('pre_vertical')[1])).to(self.device)
        norm_perm = torch.tensor(norm(self.perm, retrieve_min_max('perm_vertical')[0], retrieve_min_max('perm_vertical')[1])).to(self.device)
        norm_perm = norm_perm.expand_as(norm_sat)
        assert norm_sat.shape == norm_pre.shape == norm_perm.shape, f"Variables shapes mismatch: {norm_sat.shape}, {norm_pre.shape}, {norm_perm.shape}"
        
        if self.condition_space == 'solution':
            whole = torch.stack([norm_sat, norm_pre], dim=0)
            # whole = norm_sat
        elif self.condition_space == 'parameter':
            whole = norm_perm
        else:
            raise ValueError(f"Unknown condition space: {self.condition_space}")
        whole = whole.unsqueeze(0)
        print(f'Measurement shape: {whole.shape}')
        return whole
    
    def retrieve_reference(self):
        reference = {
            'perm': self.perm,
            'sat': self.sat,
            'pre': self.pre,
        }
        print("Retrieving reference:")
        for key, value in reference.items():
            print(f"{key}: shape {value.shape}")
        return reference
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)