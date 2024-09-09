import torch
from copy import deepcopy

def get_image_encoder(vae):
    def encode_img(input_img):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        if len(input_img.shape)<4:
            input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(input_img*2 - 1) # Note scaling
        return 0.18215 * latent.latent_dist.sample()
    return encode_img

def get_image_decoder(vae):
    def decode_img(latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach()
        return image
    return decode_img

def get_model_from_pipeline(pipeline, prompt=None, guidance_scale=0):
    if prompt is None:
        embed, negative_embed = pipeline.encode_prompt("", device=torch.device('cuda'), num_images_per_prompt=1, do_classifier_free_guidance=True)
        model = deepcopy(pipeline.unet)
        def CALL(x, t):
            return pipeline.unet(x, t, negative_embed)

        setattr(model, 'forward', CALL)
        return model
    else:
        embed, negative_embed = pipeline.encode_prompt(prompt, device=torch.device('cuda'), num_images_per_prompt=1, do_classifier_free_guidance=True)
        model = deepcopy(pipeline.unet)
        def CALL(x, t):
            prompt_out = pipeline.unet(x, t, embed)
            uncond_out = pipeline.unet(x, t, negative_embed)
            
            prompt_out.sample = uncond_out.sample + guidance_scale * (prompt_out.sample - uncond_out.sample)
            return prompt_out

        setattr(model, 'forward', CALL)
        return model

def get_diffuser_pipeline_components(model_str, scheduler_timesteps):
    if model_str == 'SD1.5':
        from diffusers import DDPMScheduler, DiffusionPipeline, PNDMScheduler
        ddpm = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        pipeline = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", scheduler=ddpm, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
        ddpm.set_timesteps(timesteps=scheduler_timesteps)
        encode_img = get_image_encoder(pipeline.vae)
        decode_img = get_image_decoder(pipeline.vae)
        model = get_model_from_pipeline(pipeline)
        
    elif model_str == 'SD2':
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
        model_id = "stabilityai/stable-diffusion-2-1-base"

        ddpm = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        ddpm.set_timesteps(timesteps=scheduler_timesteps)
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, scheduler=ddpm, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        encode_img = get_image_encoder(pipeline.vae)
        decode_img = get_image_decoder(pipeline.vae)
        model = get_model_from_pipeline(pipeline)

    else:
        raise ValueError(f"Unexpected model name: {model_str}")

    if scheduler_timesteps is not None:
        pipeline.scheduler.set_timesteps(timesteps=scheduler_timesteps)


    return model, encode_img, decode_img, ddpm

def get_finishing_scheduler(model_str, finishing_scheduler_str):
    if model_str == 'SD1.5':
        from diffusers import PNDMScheduler, DDPMScheduler
        ddpm = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        pndm = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        pndm.set_timesteps(num_inference_steps=1000)
        
    elif model_str == 'SD2':
        from diffusers import DDIMScheduler, DDPMScheduler
        model_id = "stabilityai/stable-diffusion-2-1-base"
        ddpm = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        pndm = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pndm.set_timesteps(num_inference_steps=1000)

    return pndm if finishing_scheduler_str == 'pndm' else ddpm
