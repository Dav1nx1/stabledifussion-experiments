from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionLatentUpscalePipeline,
)
import gradio as gr
import torch
from PIL import Image
import time
import psutil
import random
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from datetime import datetime
import os

def generate_folders():
  folder_name = "generated"
  folder_to_save = os.path.join(os.getcwd(), folder_name)
  # Check if the folder exists
  if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"The folder '{folder_name}' has been created.")
  else:
    print(f"The folder '{folder_name}' already exists.")
  return folder_to_save

def save_image(folder_to_save, image, name):
  current_time = datetime.now().strftime('%Y%m%d%H%M%S')
  image_path = f"{(folder_to_save)}/px_{(current_time)}_{(name)}.png"
  image.save(image_path)
  return image_path

folder_to_save = generate_folders()

start_time = time.time()
current_steps = 25
camera_definition = ', photo realistic, RAW photo, TanvirTamim, high quality, highres, sharp focus, extremely detailed, cinematic lighting, 8k uhd'
#SAFETY_CHECKER = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker", torch_dtype=torch.float16)
#prompt = "photo, 1girl, solo, long hair, breasts, looking at viewer, smile, black hair, navel, jewelry, medium breasts, nipples, standing, nude, outdoors, teeth, day, water, mole, grin, bracelet, completely nude, ring, own hands together, (Sitting with hands clasping in the knee)+++" + camera_definition
#prompt=prompt='The picture is a selfie of a woman in a home studio setting. She wears large, black-framed glasses and over-ear headphones, suggesting she may be engaged in audio work, like podcasting, music production, or streaming. Her hair is styled in two casual braids, and she wears a sleeveless top. The room is illuminated by soft lighting, and behind her is a window with blinds, a microphone with a pop filter, and various other electronic equipment. The setting appears to be a personal workspace, neatly organized, with a homey ambiance.'
#prompt="prompt='(masterpiece,  best quality,  dynamic angle),  full body shot,  (nsfw:1.4),  bend over,  kick,  hands on hips,  back view,  from below, woman,  sneakers, fit,  medium breasts,  (detailed pussy),  very thin pussy,  slit pussy,  anus,  ass, cave,  stream,  campfire,  tent,  food, dark studio,  rim lighting,  two tone lighting,  dimly lit,  low key,  Ethnic Clothing,  jewelry,  traditional clothes,  mrrpss, <lora:EMS-69762-EMS:0.600000>, , <lora:EMS-44624-EMS:0.400000>, , <lora:EMS-75843-EMS:-1.000000>, , <lora:EMS-75837-EMS:0.600000>, , <lora:EMS-75835-EMS:0.800000>, , <lora:EMS-8103-EMS:0.800000>'"
prompt="(masterpiece, best quality, dynamic angle), surrealistic, colorful, (nsfw:1.2), Thomasin McKenzie, fishnet, long sleeves, chain necklace, thigh straps, bracelet, sneakers, jewelery, gold, yellow hair, ponytail, frowning, makeup, fit, abs, large breasts, nipples, pussy, anus, sitting on the ground, arms up, teal, wall background, (detailed face, detailed eyes), looking away, God rays, hard shadow, dark studio, rim lighting, dimly lit, low key, mrrpss, <lora:EMS-179-EMS:0.600000>, , <lora:EMS-69762-EMS:0.500000>, , <lora:EMS-75851-EMS:-1.600000>, , <lora:EMS-8103-EMS:0.600000>, , <lora:EMS-75843-EMS:-0.600000>, , <lora:EMS-75835-EMS:1.400000>"
negative_prompt='text,logo,extra fingers,bad hand,age spot,watermark,(worst quality:2), (low quality:2), (normal quality:2),extra staff,extra hand,(extra arm:1.5),((multiple breasts)),((multiple tongue)),((multiple nipple)),pubic hair,blurry'
model="../../2-models/difussers/cyberrealistic_v40"
dpm = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler", solver_order=2, use_karras_sigmas=True)

pipeline = StableDiffusionPipeline.from_pretrained(
  model,
  torch_dtype=torch.float16,
  variant="fp16",
  requires_safety_checker = False,
  use_safetensors=True,
  scheduler=dpm
).to("cuda")


pipeline.enable_attention_slicing()
pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()

model_id = "stabilityai/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
upscaler.to("cuda")

#generator = torch.manual_seed(1000)
generator = torch.Generator(device="cuda")

low_res_latents = pipeline(prompt,
  generator=generator,
  output_type="latent",
  height=1216,
  width=832,
  num_inference_steps=20,
  guidance_scale=7.5,
  strength=0.6).images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

image.save("./generated/a1.png")

upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]

upscaled_image.save("./generated/a2.png")