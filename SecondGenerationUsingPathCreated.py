import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from datetime import datetime

prompt = "best quality, high resolution, (realistic:1.2), young woman, brown hair, brown eyes,Front, detailed face, beautiful eyes, (fair skin:1.2), (soft saturation:1.3) Negative prompt: BadBras,(worst quality:2),(low quality:1.4),(logo,mark:2),(undressing:1.5), (disheveled clothes:1.4),(manicure:1.2),(nipple:1.2),(long neck:2), Steps: 40, Sampler: Euler a, CFG scale: 6, Seed: 2993543141, Size: 512x768, Clip skip: 2"
negative_prompt = '(deformed iris, deformed face, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
vae_model_path = "../2-models/vae/vae-ft-mse-84-ema-pruned"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained("../2-models/difussers/cyberrealistic_v40",
  torch_dtype=torch.float16,
  safety_checker = None,
  requires_safety_checker = False,
  negative_prompt=negative_prompt,
  vae=vae,
  steps=100,
  use_safetensors=True,
  )

pipe = pipe.to("cuda")

image = pipe(prompt).images[0]
current_time = datetime.now().strftime('%Y%m%d%H%M')
image_path = f"/home/dav1nx1/Code/1-dav1nx1/1-LLM-HUGGINFACE/4-generated/images/px_{(current_time)}.png"

image.save(image_path)