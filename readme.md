# Image Compression with Stable Diffusion


## Setup

```
git clone https://github.com/anonymous-author-3/diffusion-compression.git
cd diffusion-compression
conda create env -f environment.yml
conda activate diffusion-compression
```

## Usage

```
python roundtrip.py --image_path data/kodak/23.png  --config configs/SD1.5-default.yaml --log_dir logs/SD1.5-default/kodak/23
```

## Acknowledgements

Thanks to https://github.com/danieleades/arithmetic-coding for the entropy coding library.