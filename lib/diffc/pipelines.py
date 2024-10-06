import torch

from lib.diffc.schedulers.custom_timestep_ddim import CustomTimestepDDIMScheduler
from lib.diffc.schedulers.no_variance_ddpm import NoVarianceDDPMScheduler


def get_image_encoder(vae):
    def encode_img(input_img):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(input_img*2 - 1) # Note scaling
        return vae.config.scaling_factor * latent.latent_dist.sample() # use vae.config.scaling_factor instead of this magic number
    return encode_img

def get_image_decoder(vae):
    def decode_img(latents):
        # batch of latents -> list of images
        latents = (1 / vae.config.scaling_factor) * latents # use vae.config.scaling_factor instead of this magic number
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image
    return decode_img

def get_encoding_pipeline(SD_version):
    if SD_version == 'XL':
        from diffusers import AutoPipelineForText2Image, AutoencoderKL
        model_str="stabilityai/stable-diffusion-xl-base-1.0"

        ddpmnv = NoVarianceDDPMScheduler.from_pretrained(model_str, subfolder="scheduler")

        encoding_pipeline = AutoPipelineForText2Image.from_pretrained(
            model_str, scheduler=ddpmnv, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    elif SD_version == '1.5':
        from diffusers import StableDiffusionPipeline
        model_str = "sd-legacy/stable-diffusion-v1-5"
        ddpmnv = NoVarianceDDPMScheduler.from_pretrained(model_str, subfolder="scheduler")
        encoding_pipeline = StableDiffusionPipeline.from_pretrained(model_str, scheduler=ddpmnv, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        vae = encoding_pipeline.vae

    elif SD_version == '2':
        from diffusers import DiffusionPipeline
        model_str = "stabilityai/stable-diffusion-2-1-base"
        ddpmnv = NoVarianceDDPMScheduler.from_pretrained(model_str, subfolder="scheduler")
        encoding_pipeline = DiffusionPipeline.from_pretrained(model_str, scheduler=ddpmnv, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        vae = encoding_pipeline.vae

    else:
        raise NotImplementedError(f"--SD_version {SD_version} is not a supported version yet")
    
    encode_img = get_image_encoder(vae)
    decode_img = get_image_decoder(vae)

    return encoding_pipeline, encode_img, decode_img

def get_denoising_pipeline(SD_version, encoding_pipeline):
    if SD_version == 'XL':
        model_str="stabilityai/stable-diffusion-xl-base-1.0"
        ddim_base = CustomTimestepDDIMScheduler.from_pretrained(model_str, subfolder="scheduler")
        base = DiffusionPipeline.from_pretrained(
            model_str, vae=vae, text_encoder_2=encoding_pipeline.text_encoder_2, scheduler=ddim_base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        ddim_refiner = CustomTimestepDDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="scheduler")
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=encoding_pipeline.text_encoder_2,
            vae=vae,
            scheduler=ddim_refiner,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")

    elif SD_version == '1.5':
        from diffusers import StableDiffusionPipeline
        model_str = "sd-legacy/stable-diffusion-v1-5"

        ddim_base = CustomTimestepDDIMScheduler.from_pretrained(model_str, subfolder="scheduler")
        base = StableDiffusionPipeline.from_pretrained(model_str, scheduler=ddim_base, safety_checker=None, torch_dtype=torch.float16).to("cuda")

        refiner = None

    elif SD_version == '2':
        from diffusers import DiffusionPipeline
        model_str = "stabilityai/stable-diffusion-2-1-base"

        ddim_base = CustomTimestepDDIMScheduler.from_pretrained(model_str, subfolder="scheduler")
        base = DiffusionPipeline.from_pretrained(model_str, scheduler=ddim_base, safety_checker=None, use_safetensors=True, torch_dtype=torch.float16).to("cuda")

        refiner = None
    else:
        raise NotImplementedError(f"--SD_version {SD_version} is not a supported version yet")
    
    return base, refiner
