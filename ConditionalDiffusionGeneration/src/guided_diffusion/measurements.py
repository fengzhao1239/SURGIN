'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from ConditionalNeuralField.cnf.inference_function import pass_through_model_batch
from ConditionalNeuralField.cnf.utils.normalize import Normalizer_ts
from ConditionalNeuralField.cnf.nf_networks import SIRENAutodecoder_film, SIRENAutodecoder_mdf_film
import numpy as np
from einops import rearrange
import h5py
from scipy.stats import qmc
import warnings

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

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data.to(self.device) * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)
    
@register_operator(name='case2')
class Case2Operator(NonLinearOperator):
    def __init__(self, device,
                 ckpt_path,
                 max_val,
                 min_val,
                 coords,
                 batch_size):
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)

        self.x_normalizer = Normalizer_ts(method = '-11',dim=0,
                                    params = [torch.tensor([1.,1.], device = device),
                                            torch.tensor([0.,0.], device = device)])
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, 
                                    params = [torch.tensor([[0.9617, 0.2666, 0.2869, 0.0290]], device = device), 
                                            torch.tensor([[-0.0051, -0.2073, -0.2619, -0.0419]], device = device)])
        cin_size, cout_size = 2,4
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,10,256)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size

    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...]

    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', None)
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        phy_fields = pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.x_normalizer, self.y_normalizer,
                                              self.batch_size, self.device)
        return mask*phy_fields

@register_operator(name='case3')
class Case3Operator(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                        self.x_normalizer, self.y_normalizer,
                                        self.batch_size, self.device)
        
@register_operator(name='case3_gappy')
class Case3Operator_gappy(NonLinearOperator):
    def __init__(self, device,
                 coords,
                 batch_size,
                 max_val,
                 min_val,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_ub,x_lb = params['x_normalizer_params']
        y_ub,y_lb = params['y_normalizer_params']
        cin_size, cout_size = 2,2
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_ub,x_lb))
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_ub[:cout_size],y_lb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,256,cout_size,17,256)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        out =  pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                              self.batch_size, self.x_normalizer, self.y_normalizer,
                                              self.device)
        out[:, :10, 1] = 0.
        out[:,10:, 0] = 0.
        return out

@register_operator(name='case4')
class Case4Operator(NonLinearOperator):
    def __init__(self, device,
                 coords_path,
                 batch_size,
                 max_val_path,
                 min_val_path,
                 normalizer_params_path,
                 ckpt_path
                 ) -> None:
        
        self.device = device
        coords = np.load(coords_path)
        self.coords = torch.tensor(coords, dtype = torch.float32, device=device)
        
        params = torch.load(normalizer_params_path)
        x_uub, x_llb = params['x_normalizer_params']
        y_uub,_ = params['y_normalizer0u_params']
        _,y_llb = params['y_normalizer0l_params']
        cin_size, cout_size = 3,3
        self.x_normalizer = Normalizer_ts(method = '-11',dim=0, params = (x_uub,x_llb))  # only take out xyz 
        self.y_normalizer = Normalizer_ts(method = '-11',dim=0, params = (y_uub[:cout_size],y_llb[:cout_size]))
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_film(cin_size,384,cout_size,15,384) 
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        max_val = np.load(max_val_path)
        min_val = np.load(min_val_path)
        self.max_val = torch.from_numpy(max_val).to(device) 
        self.min_val = torch.from_numpy(min_val).to(device)
        
        self.batch_size = batch_size
        
    def _unnorm(self, norm_data):
        return ((norm_data[:, 0, ...] + 1)*(self.max_val- self.min_val)/2 + self.min_val)[:, None, ...] 
    
    def forward(self, data, **kwargs):
        data_reshaped = rearrange(self._unnorm(data), "s c t l -> (s c t) l")
        return pass_through_model_batch(self.coords, data_reshaped, self.model, 
                                            self.x_normalizer, self.y_normalizer, self.batch_size,
                                              self.device)

# ====================
# Self-defined classes
# ====================


# @register_operator(name='cartesian_sparse_measurement')
# class CartesianOperatorSparse(NonLinearOperator):
#     '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
#     def __init__(self,
#                  device,
#                  ckpt_path,
#                  norm_record_path,
#                  ) -> None:
        
#         self.device = device
#         self.norm_record_path = norm_record_path
#         self.query_points = [(10, 10), (20, 5), (40, 5), (50, 15), (60, 25), (45, 2), (33, 20), (30, 40), (15, 15), (55, 32)]    # todo: specify ur measurement points
#         # self.query_points = [(10, 10), (50, 10)]
#         self.trajectory_num = 4500     # todo: specify ur reference trajectory (note 4k-5k is test set)
        
#         ckpt = torch.load(ckpt_path)
#         self.model = SIRENAutodecoder_mdf_film(2, 64, 1, 6, 64)    # todo: specify ur neural field network
#         self.model.load_state_dict(ckpt['model_state_dict'])
#         self.model.eval() 
#         self.model.to(device)
        
#         ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
#         self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
#         self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
    
#     def forward(self, data, **kwargs):
#         # ! diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>

#         traj_num = data.shape[0]
#         time_steps = data.shape[2]
#         cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
#         # ! use cnf to decode
#         cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, 2>
#         cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
#         cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>

#         cnf_out = self.model(cnf_coord_in, cnf_latent_in).squeeze()    # <(b*t), N, 1>

#         cnf_out_traj = rearrange(cnf_out, "(b t) N -> b t N", b=traj_num, t=time_steps)    # <b, t, N>

#         cnf_out_traj = self._unnorm_cnf(cnf_out_traj)
#         assert cnf_out_traj.shape == (traj_num, time_steps, len(self.query_points)), \
#             f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}), but got {cnf_out_traj.shape}"

#         return cnf_out_traj
    
#     def _unnorm(self, norm_data):
#         # * for diffusion denorm
#         return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

#     def _unnorm_cnf(self, norm_data):
#         # * for cnf denorm
#         max_val = self._get_cartesian_normalizer()["y_max"].to(self.device)
#         min_val = self._get_cartesian_normalizer()["y_min"].to(self.device)
#         return (norm_data + 1) * (max_val - min_val) / 2 + min_val
    
#     def _get_cartesian_normalizer(self):
#         # * for cnf denorm
#         normalize_records = torch.load(self.norm_record_path)
#         x_records, y_records = normalize_records["x_normalizer_params"], normalize_records["y_normalizer_params"]
#         x_max, x_min = x_records[0], x_records[1]
#         y_max, y_min = y_records[0], y_records[1]
#         return {
#             'x_max': x_max,
#             'x_min': x_min,
#             'y_max': y_max,
#             'y_min': y_min
#         }
    
#     def _gene_cartesian_coord(self):
#         # generate the Cartesian coordinates
#         h = w = 64
#         H = W = 640
#         x_coord = np.linspace(0, H, h)
#         y_coord = np.linspace(0, W, w)
#         xx, yy = np.meshgrid(x_coord, y_coord)
#         xy_coord = np.stack((xx, yy), axis=-1)
#         assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
#         xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
#         x_min, x_max = self._get_cartesian_normalizer()["x_min"], self._get_cartesian_normalizer()["x_max"]
#         xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
#         return xy_coord

#     def sparse_cartesian_coord(self):
#         '''retrieve the measuring coordinates, maybe used for visualization'''
#         xy_coord = self._gene_cartesian_coord()
#         query_pts = self.query_points    # ur query points
#         query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
#         assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
#         return query_coord
        
#     def sparse_cartesian_measurement(self):
#         '''will be used for || y - A * x ||, and visualizing the true param'''
#         # retrieve the reference dataset 
#         with h5py.File("/ehome/zhao/nf/CoNFiLD/Dataset/Cartesian.hdf5", "r") as f:
#             data = f["saturation"][:].astype(np.float32)
#             param = f["permeability_log"][:].astype(np.float32)
#         data = data[self.trajectory_num, ...]
#         param = param[self.trajectory_num, ...]
#         assert len(data.shape) == 3, f"Expected data shape (t, h, w), but got {data.shape}"
#         assert len(param.shape) == 2, f"Expected param shape (h, w), but got {param.shape}"
#         # retrieve the reference measurements
#         q_coord = self.query_points
#         measurements = torch.tensor([[data[t, i, j] for (i, j) in q_coord] for t in range(data.shape[0])], dtype=torch.float32)  # <t, N>
#         assert measurements.shape == (data.shape[0], len(q_coord)), \
#             f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
#         return {
#             'measurements': measurements,
#             'reference_data': data,
#             'reference_param': param,
#         }
        

# @register_operator(name='cartesian_Real_superresolution')
# class CartesianOperatorSR(NonLinearOperator):
#     '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
#     def __init__(self,
#                  device,
#                  ckpt_path,
#                  norm_record_path,
#                  ds_size,
#                  ) -> None:
        
#         self.device = device
#         self.norm_record_path = norm_record_path
#         self.ds_size = ds_size   # todo: specify ur measurement points
#         self.trajectory_num = 4100     # todo: specify ur reference trajectory (note 4k-5k is test set)
        
#         ckpt = torch.load(ckpt_path)
#         self.model = SIRENAutodecoder_mdf_film(2, 64, 1, 6, 64)    # todo: specify ur neural field network
#         self.model.load_state_dict(ckpt['model_state_dict'])
#         self.model.eval() 
#         self.model.to(device)
        
#         ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
#         self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
#         self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
    
#     def forward(self, data, **kwargs):
#         # ! diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>
#         traj_num = data.shape[0]
#         time_steps = data.shape[2]
#         cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
#         # ! use cnf to decode
#         cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <h, w, 2>
#         cnf_latent_in = cnf_latent_in.unsqueeze(1).unsqueeze(1)    # <(b*t), 1, 1, l>
#         cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, h, w, 2>
#         assert len(cnf_latent_in.shape) == len(cnf_coord_in.shape) == 4, \
#             f"CNF Decoder input shape [{cnf_coord_in.shape}, {cnf_latent_in.shape}] mismatch, expected 4D tensor"

#         cnf_out = self.model(cnf_coord_in, cnf_latent_in).squeeze(-1)    # <(b*t), h, w, 1>

#         cnf_out_traj = rearrange(cnf_out, "(b t) h w -> b t h w", b=traj_num, t=time_steps)    # <b, t, h, w>

#         cnf_out_traj = self._unnorm_cnf(cnf_out_traj)
#         assert cnf_out_traj.shape == (traj_num, time_steps, 64, 64), \
#             f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, 64, 64), but got {cnf_out_traj.shape}"

#         down_cnf_out_traj = F.interpolate(cnf_out_traj, size=self.ds_size, mode='nearest')
#         return down_cnf_out_traj
    
#     def _unnorm(self, norm_data):
#         # * for diffusion denorm
#         return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

#     def _unnorm_cnf(self, norm_data):
#         # * for cnf denorm
#         max_val = self._get_cartesian_normalizer()["y_max"].to(self.device)
#         min_val = self._get_cartesian_normalizer()["y_min"].to(self.device)
#         return (norm_data + 1) * (max_val - min_val) / 2 + min_val
    
#     def _get_cartesian_normalizer(self):
#         # * for cnf denorm
#         normalize_records = torch.load(self.norm_record_path)
#         x_records, y_records = normalize_records["x_normalizer_params"], normalize_records["y_normalizer_params"]
#         x_max, x_min = x_records[0], x_records[1]
#         y_max, y_min = y_records[0], y_records[1]
#         return {
#             'x_max': x_max,
#             'x_min': x_min,
#             'y_max': y_max,
#             'y_min': y_min
#         }
    
#     def _gene_cartesian_coord(self):
#         # generate the Cartesian coordinates
#         h = w = 64
#         H = W = 640
#         x_coord = np.linspace(0, H, h)
#         y_coord = np.linspace(0, W, w)
#         xx, yy = np.meshgrid(x_coord, y_coord)
#         xy_coord = np.stack((xx, yy), axis=-1)
#         assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
#         xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
#         x_min, x_max = self._get_cartesian_normalizer()["x_min"], self._get_cartesian_normalizer()["x_max"]
#         xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
#         return xy_coord

#     def sparse_cartesian_coord(self):
#         '''retrieve the measuring coordinates, maybe used for visualization'''
#         xy_coord = self._gene_cartesian_coord()
#         return xy_coord
        
#     def sparse_cartesian_measurement(self):
#         '''will be used for || y - A * x ||, and visualizing the true param'''
#         # retrieve the reference dataset 
#         with h5py.File("/ehome/zhao/nf/CoNFiLD/Dataset/Cartesian.hdf5", "r") as f:
#             data = f["saturation"][:].astype(np.float32)
#             param = f["permeability_log"][:].astype(np.float32)
#         data = data[self.trajectory_num, ...]
#         param = param[self.trajectory_num, ...]
#         assert len(data.shape) == 3, f"Expected data shape (t, h, w), but got {data.shape}"
#         assert len(param.shape) == 2, f"Expected param shape (h, w), but got {param.shape}"
#         # retrieve the reference measurements
#         measurements = torch.tensor(data, dtype=torch.float32)  # <t, h, w>
#         ds_measurements = F.interpolate(measurements.unsqueeze(0), size=self.ds_size, mode='nearest').squeeze(0)    # <t, ds_h, ds_w>
#         # assert ds_measurements.shape == (data.shape[0], self.ds_size, self.ds_size), \
#         #     f"Expected measurements shape ({data.shape[0]}, {self.ds_size}, {self.ds_size}), but got {ds_measurements.shape}"
#         return {
#             'measurements': ds_measurements,
#             'reference_data': data,
#             'reference_param': param,
#         }


# ===================================================================
# Self-defined classes for joint generation of solution and parameter
# ===================================================================

# ------------------------- sparse measurement -------------------------

@register_operator(name='cartesian_sparse_measurement')
class CartesianOperatorSM(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                num_probed,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        # self.query_points = self._gene_random_idx(num_probed)   # todo: specify ur measurement points
        if num_probed == 1:
            self.query_points = [(32, 28)]
        elif num_probed == 3:
            self.query_points = [(20, 20), (40, 30), (30, 50)]
        elif num_probed == 5:
            self.query_points = [(10, 10), (20, 55), (28, 26), (40, 46), (55, 15)]
        else:
            self.query_points = self._gene_random_idx(num_probed)
            warnings.warn(f"Randomly generated {num_probed} points for measurement, please check the results.")
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        # with h5py.File(simdata_path, "r") as f:
        #     # ! only apply to cartesian data
        #     pressure = f["pressure"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     saturation = f["saturation"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     param = f["permeability_log"][simdata_idx, None, :, :-1, None].astype(np.float32)
        # param = np.repeat(param, saturation.shape[1], axis=1)
        # self.simdata = np.concatenate([pressure, saturation, param], axis=-1)    # <t, h, w, c=3>
        # self.simdata = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in).squeeze(1)    # <(b*t), N, c=3> 'we add None @ dim=1'
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"

        output = cnf_out_traj[..., :-1]    # <b, t, N, c=2>
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_random_idx(self, num_pts, max_range=(63, 63)):
        SAMPLER = qmc.LatinHypercube(d=2, seed=42)
        sample = SAMPLER.random(n=num_pts)  # shape: (num_pts, 2), range: [0, 1)
        scaled = qmc.scale(sample, [0, 0], [max_range[0], max_range[1]])
        idxes = [tuple(map(int, pt)) for pt in scaled]
        return idxes

    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        # h, w = 64, 63
        # H, W = 640, 630
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||, and visualizing the true param'''
        data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.stack([torch.stack([data[t, i, j, :-1] for (i, j) in q_coord]) for t in range(data.shape[0])])  # <t, N, c=?>
        # assert measurements.shape == (data.shape[0], len(q_coord)), \
        #     f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
        print(f"Measurement shape: {measurements.shape}")
        return measurements.unsqueeze(0).to(self.device)

@register_operator(name='cartesian_sparse_measurement_all')
class CartesianOperatorSMAll(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                num_probed,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        # self.query_points = self._gene_random_idx(num_probed)   # todo: specify ur measurement points
        if num_probed == 1:
            self.query_points = [(32, 28)]
        elif num_probed == 3:
            self.query_points = [(20, 20), (40, 30), (30, 50)]
        elif num_probed == 5:
            self.query_points = [(10, 10), (20, 55), (28, 26), (40, 46), (55, 15)]
        else:
            self.query_points = self._gene_random_idx(num_probed)
            warnings.warn(f"Randomly generated {num_probed} points for measurement, please check the results.")
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        # with h5py.File(simdata_path, "r") as f:
        #     # ! only apply to cartesian data
        #     pressure = f["pressure"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     saturation = f["saturation"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     param = f["permeability_log"][simdata_idx, None, :, :-1, None].astype(np.float32)
        # param = np.repeat(param, saturation.shape[1], axis=1)
        # self.simdata = np.concatenate([pressure, saturation, param], axis=-1)    # <t, h, w, c=3>
        # self.simdata = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in).squeeze(1)    # <(b*t), N, c=3> 'we add None @ dim=1'
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"

        output = cnf_out_traj[..., :]    # <b, t, N, c=2>
        return output
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_random_idx(self, num_pts, max_range=(63, 63)):
        SAMPLER = qmc.LatinHypercube(d=2, seed=42)
        sample = SAMPLER.random(n=num_pts)  # shape: (num_pts, 2), range: [0, 1)
        scaled = qmc.scale(sample, [0, 0], [max_range[0], max_range[1]])
        idxes = [tuple(map(int, pt)) for pt in scaled]
        return idxes

    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        # h, w = 64, 63
        # H, W = 640, 630
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||, and visualizing the true param'''
        data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.stack([torch.stack([data[t, i, j, :] for (i, j) in q_coord]) for t in range(data.shape[0])])  # <t, N, c=?>
        # assert measurements.shape == (data.shape[0], len(q_coord)), \
        #     f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
        print(f"Measurement shape: {measurements.shape}")
        return measurements.unsqueeze(0).to(self.device)


# ------------------------- low fidelity measurement (super-resolution) -------------------------

@register_operator(name='cartesian_superresolution')
class CartesianOperatorSR(NonLinearOperator):
    '''for sparse measurement, we only need to query the measured predicted points A*x, and provide the measured data for y'''
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                ds_size,
                vanilla_flag=False,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        self.ds_size = (ds_size, ds_size) if isinstance(ds_size, int) else tuple(ds_size)    # todo: specify ur downsampling size
        self.vanilla_flag = vanilla_flag
        if vanilla_flag:
            warnings.warn(f"Vanilla flag is set to True, using the original data without downsampling!")
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=5,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        # with h5py.File(simdata_path, "r") as f:
        #     # ! only apply to cartesian data
        #     pressure = f["pressure"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     saturation = f["saturation"][simdata_idx, :, :, :-1, None].astype(np.float32)
        #     param = f["permeability_log"][simdata_idx, None, :, :-1, None].astype(np.float32)
        # param = np.repeat(param, saturation.shape[1], axis=1)
        # self.simdata = np.concatenate([pressure, saturation, param], axis=-1)    # <t, h, w, c=3>
        # self.simdata = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        # print(f"Simulation data loaded, with shape {self.simdata.shape}")
        self.simdata = np.load(simdata_path)[simdata_idx]
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <h, w, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1).unsqueeze(1)    # <(b*t), 1, 1 l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, h, w, 2>
        assert len(cnf_latent_in.shape) == len(cnf_coord_in.shape) == 4, \
            f"CNF Decoder input shape [{cnf_coord_in.shape}, {cnf_latent_in.shape}] mismatch, expected 4D tensor"
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), h, w, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) h w c-> b t h w c", b=traj_num, t=time_steps)    # <b, t, h, w, c>
        # cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:2] == (traj_num, time_steps), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, H, W, C), but got {cnf_out_traj.shape}"
        
        # * downsampling
        cnf_out_traj_reshape = rearrange(cnf_out_traj, "b t h w c -> (b t) c h w")
        if self.vanilla_flag:
            down_cnf_out_traj = cnf_out_traj_reshape
        else:
            down_cnf_out_traj = F.interpolate(cnf_out_traj_reshape, size=self.ds_size, mode='nearest')
        output = rearrange(down_cnf_out_traj, "(b t) c h w -> b t h w c", b=traj_num, t=time_steps)
        # 0:1 -- pressure
        # 1:2 -- saturation
        # 2:3 -- permeability
        return output[:, :, ..., 0:2]    # <b, t, h, w, c=1>
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _norm_cnf(self, raw_data):
        # * for cnf norm
        normed_data = self.out_normalizer.normalize(raw_data)
        return normed_data
    
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        # h, w = 64, 63
        # H, W = 640, 630
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        return xy_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||'''
        # data = self.simdata   # <t, h, w, c=3>
        data = self._norm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        # data = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        data_reshape = rearrange(data, "t h w c -> t c h w")
        if self.vanilla_flag:
            ds_measurements = data_reshape
        else:
            ds_measurements = F.interpolate(data_reshape, size=self.ds_size, mode='nearest')    # <t, c, ds_h, ds_w>
        out_measurements = rearrange(ds_measurements, "t c h w -> t h w c")
        return out_measurements.unsqueeze(0).to(self.device)[:, :, ..., 0:2]    # <1, t, h, w, c=1>


# ------------------------- damaged measurement -------------------------

@register_operator(name='cartesian_damaged_measurement')
class CartesianOperatorDM(NonLinearOperator):
    def __init__(self,
                device,
                ckpt_path,
                norm_record_path,
                simdata_path,
                simdata_idx,
                top_left_idx,
                bottom_right_idx,
                ) -> None:
        
        self.device = device
        self.norm_record_path = norm_record_path
        assert isinstance(top_left_idx, tuple) and isinstance(bottom_right_idx, tuple), \
            f"Expected top_left_idx and bottom_right_idx to be tuples, but got {type(top_left_idx)} and {type(bottom_right_idx)}"
        self.query_points = self._generate_query_idx(top_left_idx, bottom_right_idx)    # todo: specify ur damaged area
        
        ckpt = torch.load(ckpt_path)
        self.model = SIRENAutodecoder_mdf_film(omega_0=10,
                                                in_coord_features=2,
                                                in_latent_features=256,
                                                out_features=3,
                                                num_hidden_layers=5,
                                                hidden_features=128)    # todo: specify ur neural field network
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval() 
        self.model.to(device)
        
        ckpt_latent = ckpt["hidden_states"]["latents"]    # <n*t, dims>
        self.latent_max, self.latent_min = torch.max(ckpt_latent), torch.min(ckpt_latent)
        self.latent_max, self.latent_min = self.latent_max.to(self.device), self.latent_min.to(self.device)
        
        normalize_records = torch.load(self.norm_record_path)
        self.out_normalizer = Normalizer_ts(method="-11", dim=0)
        self.out_normalizer.params = normalize_records["y_normalizer_params"]
        
        with h5py.File(simdata_path, "r") as f:
            # ! only apply to cartesian data
            pressure = f["pressure"][simdata_idx, :, :, :-1, None].astype(np.float32)
            saturation = f["saturation"][simdata_idx, :, :, :-1, None].astype(np.float32)
            param = f["permeability_log"][simdata_idx, None, :, :-1, None].astype(np.float32)
        param = np.repeat(param, saturation.shape[1], axis=1)
        self.simdata = np.concatenate([pressure, saturation, param], axis=-1)    # <t, h, w, c=3>
        self.simdata = torch.tensor(self.simdata, dtype=torch.float32).to(self.device)    # <t, h, w, c=3>
        print(f"Simulation data loaded, with shape {self.simdata.shape}")
        
    def forward(self, data, **kwargs):
        # * diffusion data: <b, c=1, t, l>  --> cnf in: <b*c*t, l>, unnorm
        traj_num = data.shape[0]
        time_steps = data.shape[2]
        cnf_latent_in = rearrange(self._unnorm(data), "b c t l -> (b c t) l").to(self.device)    # <(b*t), l>
        
        # * use cnf to decode
        cnf_coord_in = self.sparse_cartesian_coord().to(self.device)    # <N, coord_dim=2>
        cnf_latent_in = cnf_latent_in.unsqueeze(1)    # <(b*t), 1, l>
        cnf_coord_in = cnf_coord_in.unsqueeze(0)    # <1, N, 2>
        cnf_out = self.model(cnf_coord_in, cnf_latent_in)    # <(b*t), N, c=3>
        
        # * reshape and unnorm
        cnf_out_traj = rearrange(cnf_out, "(b t) N c-> b t N c", b=traj_num, t=time_steps)    # <b, t, N, c>
        cnf_out_traj = self._unnorm_cnf(cnf_out_traj) #! change
        assert cnf_out_traj.shape[:-1] == (traj_num, time_steps, len(self.query_points)), \
            f"Expected cnf_out_traj shape ({traj_num}, {time_steps}, {len(self.query_points)}, C), but got {cnf_out_traj.shape}"

        return cnf_out_traj
    
    def _unnorm(self, norm_data):
        # * for diffusion unnorm
        return (norm_data + 1) * (self.latent_max - self.latent_min) / 2 + self.latent_min

    def _unnorm_cnf(self, norm_data):
        # * for cnf unnorm
        unnormed_data = self.out_normalizer.denormalize(norm_data)
        return unnormed_data
    
    def _generate_query_idx(self, top_left_idx, bottom_right_idx):
        i1, j1 = top_left_idx
        i2, j2 = bottom_right_idx
        ii, jj = np.meshgrid(np.arange(i1, i2), np.arange(j1, j2), indexing='ij')
        query_idx = list(zip(ii.flatten(), jj.flatten()))  # list of (i, j)
        return query_idx
        
    def _gene_cartesian_coord(self):
        # generate the Cartesian coordinates
        h = w = 64
        H = W = 640
        # h, w = 64, 63
        # H, W = 640, 630
        x_coord = np.linspace(0, H, h)
        y_coord = np.linspace(0, W, w)
        xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
        xy_coord = np.stack((xx, yy), axis=-1)
        assert xy_coord.shape == (h, w, 2), f"Expected coord shape (h, w, 2), but got {xy_coord.shape}"
        xy_coord = torch.tensor(xy_coord, dtype=torch.float32)
        x_min, x_max = xy_coord.min(), xy_coord.max()
        xy_coord = ((xy_coord - x_min) / (x_max - x_min)) * 2 - 1
        return xy_coord

    def sparse_cartesian_coord(self):
        '''retrieve the measuring coordinates, maybe used for visualization'''
        xy_coord = self._gene_cartesian_coord()
        query_pts = self.query_points    # ur query points
        query_coord = torch.stack([xy_coord[i, j] for (i, j) in query_pts], dim=0).float()    # <N, 2>
        assert query_coord.shape == (len(query_pts), 2), f"Expected query coord shape ({len(query_pts)}, 2), but got {query_coord.shape}"
        return query_coord
        
    def sparse_cartesian_measurement(self):
        '''will be used for || y - A * x ||, and visualizing the true param'''
        # data = self.simdata    # <t, h, w, c=3>
        data = self._unnorm_cnf(torch.tensor(self.simdata))    # <t, h, w, c=3>
        q_coord = self.query_points
        measurements = torch.tensor([[data[t, i, j, :-1] for (i, j) in q_coord] for t in range(data.shape[0])], dtype=torch.float32)  # <t, N, c=?>
        assert measurements.shape == (data.shape[0], len(q_coord)), \
            f"Expected measurements shape ({data.shape[0]}, {len(q_coord)}), but got {measurements.shape}"
        return measurements.unsqueeze(0).to(self.device)


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