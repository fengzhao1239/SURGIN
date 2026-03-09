## Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import sys
import numpy as np
from src.script_util import create_model, create_gaussian_diffusion, create_DiT_model
from basicutility import ReadInput as ri
from latents.create_dataset import Multi_Cartesian_Dataset

## Setup
if torch.cuda.is_available():  
  dev = "cuda" 
else:  
  dev = "cpu"
  
device = torch.device(dev)  

torch.manual_seed(42)
np.random.seed(42)

inp = ri.basic_input(sys.argv[1])

## Hyperparams
test_batch_size = inp.test_batch_size # num of samples to generate

image_size= inp.image_size
# num_channels= inp.num_channels
# num_res_blocks= inp.num_res_blocks
# num_heads= inp.num_heads
# num_head_channels = inp.num_head_channels
# attention_resolutions = inp.attention_resolutions
# channel_mult = inp.channel_mult

patch_size = inp.patch_size
in_channels = inp.in_channels
hidden_size = inp.hidden_size
depth = inp.depth
num_heads = inp.num_heads

steps= inp.steps
noise_schedule= inp.noise_schedule

## Create model and diffusion
# unet_model = create_model(image_size=image_size,
#                           num_channels= num_channels,
#                           num_res_blocks= num_res_blocks,
#                           num_heads=num_heads,
#                           num_head_channels=num_head_channels,
#                           attention_resolutions=attention_resolutions,
#                           channel_mult=channel_mult,
#                           learn_sigma=False, # !
#                         )
unet_model = create_DiT_model(
                            input_size=image_size,
                            patch_size=patch_size,
                            in_channels=in_channels,
                            hidden_size=hidden_size,
                            depth=depth,
                            num_heads=num_heads,
                            mlp_ratio=4.0,
                            class_dropout_prob=0.0,
                            num_classes=None,
                            learn_sigma=True,
                        )

unet_model.load_state_dict(torch.load(inp.ema_path))
unet_model.to(device)

diff_model = create_gaussian_diffusion(steps=steps,
                                      noise_schedule=noise_schedule,
                                      learn_sigma=True # !
                                    )

## Unconditional sample
print("Sampling unconditional images...")
sample_fn = diff_model.p_sample_loop
gen = sample_fn(unet_model, (4, 1, image_size, image_size))[:, 0]
print(f"Diffusion generated images have shape: {gen.shape}")

## Denormalizing the latents (load the max and min of your training latent data)
my_dataset = Multi_Cartesian_Dataset(inp.hdf5_path)

max_val, min_val = my_dataset.max, my_dataset.min
gen = (gen + 1)*(max_val - min_val)/2. + min_val
save_gene = gen.detach().cpu().numpy()
np.save(inp.save_gen_path, save_gene)
