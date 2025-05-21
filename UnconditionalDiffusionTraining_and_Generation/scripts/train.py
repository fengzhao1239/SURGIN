#Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import torch
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion
from src.train_util import TrainLoop
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from src.dist_util import setup_dist, dev
from src.logger import configure, log
from basicutility import ReadInput as ri
from latents.create_dataset import Multi_Cartesian_Dataset

## Setup
torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

setup_dist()
# setup logger system
configure(dir=inp.log_path, format_strs=["stdout","log","neptune"], neptune_tag=inp.neptune_tag)

## HyperParams (Change according to the case)
batch_size = inp.batch_size
test_batch_size = inp.test_batch_size

image_size= inp.image_size
num_channels= inp.num_channels
num_res_blocks= inp.num_res_blocks
num_heads=inp.num_heads
num_head_channels= inp.num_head_channels
attention_resolutions= inp.attention_resolutions
channel_mult = inp.channel_mult

steps= inp.steps
noise_schedule= inp.noise_schedule

microbatch= inp.microbatch
lr = inp.lr
ema_rate= inp.ema_rate
log_interval= inp.log_interval
save_interval= inp.save_interval
lr_anneal_steps= inp.lr_anneal_steps

log("-- training start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
## Data Preprocessing
log("-- creating data loader...")
norm_train_data, norm_valid_data = Multi_Cartesian_Dataset(inp.hdf5_path).create_dataset()

dl_train = DataLoader(TensorDataset(norm_train_data), batch_size=batch_size, shuffle=True)
dl_valid = DataLoader(TensorDataset(norm_valid_data), batch_size=test_batch_size, shuffle=True)

# use yield to infinitely iterate over the dataloader
def dl_iter(dl):
    while True:
        yield from dl 

## Unet Model
log("-- creating model and diffusion...")

unet_model = create_model(image_size=image_size,
                          num_channels= num_channels,
                          num_res_blocks= num_res_blocks,
                          num_heads=num_heads,
                          num_head_channels=num_head_channels,
                          attention_resolutions=attention_resolutions,
                          channel_mult=channel_mult,
                          learn_sigma=False, # !
                        )

unet_model.to(dev())
log(f"-- model created, total params: {sum(p.numel() for p in unet_model.parameters())}")

## Gaussian Diffusion
diff_model = create_gaussian_diffusion(steps=steps,
                                       noise_schedule=noise_schedule,
                                       learn_sigma=False # !
                                    )

## Training Loop
log("-- training...")

train_uncond_model = TrainLoop(
                                model=unet_model,
                                diffusion=diff_model,
                                train_data = dl_iter(dl_train),
                                valid_data=dl_iter(dl_valid),
                                batch_size= batch_size,
                                microbatch= microbatch,
                                lr = lr,
                                ema_rate=ema_rate,
                                log_interval=log_interval,
                                save_interval=save_interval,
                                lr_anneal_steps=lr_anneal_steps,
                                resume_checkpoint="",)

train_uncond_model.run_loop()