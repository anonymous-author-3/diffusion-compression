import numpy as np
from lib.DDPM_compression import zipf_rcc_encode_image
import lib.compression_utils as utils
from PIL import Image
from lib.diffusion_models import get_diffuser_pipeline_components
import yaml
import argparse
from zipf_encoding import encode_zipf 
from pathlib import Path

parser = argparse.ArgumentParser(description="Compress an image using a diffusion model.")

parser.add_argument(
    '--config_path',
    type=str,
    required=True,
    help="Path to the compression config .yaml file. Specifies details of how the image is encoded/decoded")

parser.add_argument(
    "--image_path",
    type=str,
    default='data/kodak/01.png',
    required=True,
    help="Path to the image to compress"
)

parser.add_argument(
    "--encoding_path",
    type=str,
    required=True,
    help="file path to save the encoded image to."
)

args = parser.parse_args()
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

model, encode_img, decode_img, ddpm_scheduler = get_diffuser_pipeline_components(config['model_str'], config['scheduler_timesteps'])

image = Image.open(args.image_path)
image_np = np.array(image)
image_pt = utils.np_to_torch_img(image_np)
encoded_gt = encode_img(image_pt)

(
    seeds,
    zipf_s_vals,
    zipf_n_vals
) = zipf_rcc_encode_image(
    encoded_gt,
    model,
    ddpm_scheduler,
    max_chunk_size=config['max_chunk_size'],
    chunk_padding_bits=config['chunk_padding_bits'],
    sample_shape=encoded_gt.shape,
    D_kl_per_step=config['D_kl_per_step'])

encoded_bytes = encode_zipf(zipf_s_vals, zipf_n_vals, seeds)

encoding_path = Path(args.encoding_path)
encoding_path.parent.mkdir(parents=True, exist_ok=True)

utils.save_bytes(encoded_bytes, encoding_path)
