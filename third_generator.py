import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler
from datetime import datetime

## Configuration Variables
prompt = "best quality, high resolution, (realistic:1.2), young woman, black hair, blue eyes,Front, detailed face, beauty eyes, (fair skin:1.2), (soft saturation:1.3), full body"
negative_prompt = 'BadBras,(worst quality:2),(low quality:1.4),(logo,mark:2),(undressing:1.5), (disheveled clothes:1.4),(manicure:1.2),(nipple:1.2),(long neck:2), Steps: 40, Sampler: Euler a, CFG scale: 6, Seed: 2993543141, Size: 512x768, Clip skip: 2'
vae_model_path = "../2-models/vae/vae-ft-mse-84-ema-pruned"
model="../2-models/difussers/juggernautXL_"


# Forward embeddings and negative embeddings through text encoder


## Setting the model
# enable requires_safety_checker for SFW

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(model,
  torch_dtype=torch.float16,
  safety_checker = None,
  requires_safety_checker = False,
  negative_prompt=negative_prompt,
  vae=vae,
  use_safetensors=True,
  )

pipe.to('cuda')

max_length = pipe.tokenizer.model_max_length
input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

negative_ids = pipe.tokenizer("", truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids
negative_ids = negative_ids.to("cuda")

concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

## Settings the torch seed
image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=20).images[0]

## Preparing final file and download
current_time = datetime.now().strftime('%Y%m%d%H%M')
image_path = f"/home/dav1nx1/Code/1-dav1nx1/1-LLM-HUGGINFACE/4-generated/images/px_{(current_time)}.png"

## Save in path
image.save(image_path)