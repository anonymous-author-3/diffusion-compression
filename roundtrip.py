import torch
import numpy as np
import argparse
import torch
import yaml
from easydict import EasyDict as edict
from lib import image_utils
from lib.diffc.pipelines import get_encoding_pipeline, get_denoising_pipeline
from lib.diffc.decode import SD_decode
from lib.diffc.encode import SD_encode
from lib.diffc.denoise import denoise
from lib.diffc.simulate_ideal_rcc import simulate_ideal_coding
from pathlib import Path
from PIL import Image

#####################################################################
## Arguments
#####################################################################

parser = argparse.ArgumentParser("Compress and decompress an image at multiple bitrates using Stable Diffusion, and evaluate the results.")

parser.add_argument(
    '--log_dir',
    type=str,
    help="Directory to save compression results to."
)

parser.add_argument(
    '--config',
    type=str,
    required=True,
    help="Path to the compression config .yaml file. Specifies details of how the image is encoded/decoded")

parser.add_argument(
    "--image_path",
    type=str,
    default='data/kodak/23.png',
    help="Path to the image to compress."
)

parser.add_argument(
    "--prompt",
    type=str,
    default="",
    help="Prompt to use to encode/decode and denoise the image."
)

args = parser.parse_args()

with open(args.config, 'r') as file:
    config = edict(yaml.safe_load(file))

#####################################################################
## Load Diffusion Pipelines 
#####################################################################

encoding_pipeline, encode_img, decode_img = get_encoding_pipeline(config.SD_version)

#####################################################################
## Load Image
#####################################################################

gt_np = np.array(Image.open(args.image_path))
gt_pt = image_utils.np_to_torch_img(gt_np)
n, c, h, w = gt_pt.shape
num_pixels = h*w
gt_encoded = encode_img(gt_pt)
gt_encoded

prompt = args.prompt

#####################################################################
## Compress And  Decompress
#####################################################################

encoding_timesteps = config.scheduler_timesteps
recon_timesteps = config.recon_timesteps

if config.theoretical_optimum:
    D_kl_per_step, recon_latents, recon_step_indices = simulate_ideal_coding(
        encoding_pipeline, 
        prompt,
        config.encoding_guidance_scale,
        gt_encoded,
        timesteps=encoding_timesteps,
        recon_steps=recon_timesteps)
    
    # Dummy values for all these unused stats
    seed_tuples = []
    computed_chunk_sizes = []
    zipf_s_vals = []
    zipf_n_vals = []
else:
    D_kl_per_step, seed_tuples, computed_chunk_sizes, zipf_s_vals, zipf_n_vals = SD_encode(
            encoding_pipeline,
            prompt,
            config.encoding_guidance_scale,
            target_image=gt_encoded,
            timesteps=encoding_timesteps,
            max_chunk_size=config.max_chunk_size,
            chunk_padding_bits=config.chunk_padding_bits,
            D_kl_per_step=config.D_kl_per_step)

    recon_latents, recon_step_indices = SD_decode(
        encoding_pipeline,
        prompt,
        config.encoding_guidance_scale,
        seed_tuples, 
        computed_chunk_sizes,
        recon_timesteps,
        gt_encoded.shape,
        timesteps=encoding_timesteps)

base, refiner = get_denoising_pipeline(config.SD_version, encoding_pipeline)
del encoding_pipeline

denoising_timesteps = config.denoising_timesteps or encoding_timesteps
reconstructions = []
for recon_latent, recon_step_index in zip(recon_latents, recon_step_indices):
    recon_timestep = int(encoding_timesteps[recon_step_index])
    recon_pil = denoise(recon_latent, recon_timestep, base, refiner, prompt, guidance_scale=config.denoising_guidance_scale, timesteps=torch.tensor(denoising_timesteps))
    reconstruction = image_utils.np_to_torch_img(np.array(recon_pil))
    reconstructions.append(reconstruction)

del encode_img, decode_img, base, refiner

#####################################################################
## Evaluate
#####################################################################

DKL_per_pixel_per_step = np.array(D_kl_per_step) / num_pixels

from lib import metrics
import sys
import zlib

if config.encoding_guidance_scale == 0 and config.denoising_guidance_scale == 0:
    prompt_bpp = 0
else:
    prompt_bpp = sys.getsizeof(zlib.compress(prompt.encode()))*8 / num_pixels

recon_bpps = []
recon_psnrs = []
recon_lpips = []
recon_clips = []
for recon, recon_step_index in zip(reconstructions, recon_step_indices):
    if config.theoretical_optimum:
        recon_bpps.append(prompt_bpp + np.sum(DKL_per_pixel_per_step[:recon_step_index+1]))
    else:
        recon_bpps.append(prompt_bpp + metrics.get_bpp(seed_tuples, zipf_s_vals, zipf_n_vals, recon_step_index, num_pixels))
    recon_psnrs.append(metrics.get_psnr(recon, gt_pt))
    recon_lpips.append(metrics.get_lpips(recon, gt_pt))
    recon_clips.append(metrics.get_clip_score(recon, gt_pt))

#####################################################################
## Save results to mongodb                                         ##
#####################################################################

# Create directories
log_dir = Path(args.log_dir)
recon_dir = log_dir / 'reconstructions'
recon_dir.mkdir(parents=True, exist_ok=True)

# Save reconstructions
recon_filenames = []
for idx, recon in enumerate(reconstructions):
    filename = f'{str(encoding_timesteps[idx]).zfill(3)}.png'
    full_path = recon_dir / filename
    image_utils.torch_to_pil_img(recon).save(full_path)
    recon_filenames.append(filename)

# Prepare data for YAML file
data = {
    "config": vars(args),
    "seed_tuples": [tuple(map(int, seed_tuple)) for seed_tuple in seed_tuples],
    "computed_chunk_sizes": computed_chunk_sizes,
    "DKL_per_pixel_per_step": [float(x) for x in DKL_per_pixel_per_step],
    "recon_steps": [encoding_timesteps[idx] for idx in recon_step_indices],
    "recon_psnrs": [float(x) for x in recon_psnrs],
    "recon_lpips": [float(x) for x in recon_lpips],
    "recon_bpps": [float(x) for x in recon_bpps],
    "recon_clip_scores": [float(x) for x in recon_clips],
    "recon_image_filenames": recon_filenames
}

# Write YAML file
yaml_path = log_dir / 'results.yaml'
yaml_path.write_text(yaml.dump(data))
