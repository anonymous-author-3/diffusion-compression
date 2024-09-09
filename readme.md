# Image Compression with Stable Diffusion

## Setup

```
git clone TODO/url.git
cd diffusion-compression
conda create env -f environment.yml
conda activate diffusion-compression
```

## Usage


To compress an image, run:

```
python compress.py --config configs/SD1.5-4.5KB.yaml --image_path data/ground_truth/kodak/01.png --encoding_path data/compressed/SD1.5-4.5/kodak/01.ddpm
```

Then to decompress that image, run:

```
python decompress.py --config configs/SD1.5-4.5KB.yaml --encoding_path data/compressed/SD1.5-4.5/kodak/01.ddpm --image_path reconstructions/SD1.5-4.5/kodak/01.png 
```

You must use the same config file to decompress an image as was used to compress it. Otherwise, the resulting reconstruction may be randomized.