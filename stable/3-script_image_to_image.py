import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, AutoencoderKL, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from datetime import datetime
from transformers import CLIPImageProcessor, CLIPModel
from PIL import ImageTk, Image
import os
from diffusers.utils import load_image, make_image_grid

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

def initialize_environment():
  torch.cuda.empty_cache()


initialize_environment()
folder_to_save = generate_folders()

prompt='complex 3d render ultra detailed of a beautiful porcelain profile woman android face (black eyes)1.8, cyborg, robotic parts, 150 mm, beautiful studio soft light, rim light, vibrant details, luxurious cyberpunk, lace, hyperrealistic, anatomical, facial muscles, cable electric wires, microchip, elegant, beautiful background, octane render, H. R. Giger style, 8k, best quality, masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,wallpaper, (realistic, photo-realistic:1.37),Amazing, finely detail, masterpiece,best quality,official art, extremely detailed CG unity 8k wallpaper, absurdres, incredibly absurdres, robot, silver halmet, full body, sitting'
negative_prompt='not-solid-best-physical-appearance-perfect::2 (disfigured:1.3), (bad art:1.3), (deformed:1.3),(extra limbs:1.3),(close up:1.3),(b&w:1.3), weird colors, blurry, (duplicate:1.5), (morbid:1.3), (mutilated:1.3), [out of frame], extra fingers, mutated hands, (poorly drawn hands:1.3), (poorly drawn face:1.3), (mutation:1.5), (deformed:1.5), (ugly:1.3), blurry, (bad anatomy:1.3), (bad proportions:1.5), (extra limbs:1.3), cloned face, (disfigured:1.5), out of frame,  (malformed limbs:1.1), (missing arms:1.3), (missing legs:1.3), (extra arms:1.5), (extra legs:1.5), mutated hands, (fused fingers:1.1), (too many fingers:1.1), (long neck:1.5), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy'

vae_model_path = "../../2-models/vae/vae-ft-mse-84-ema-pruned"
model="../../2-models/difussers/beautifulRealistic_v7"

dpm = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler", solver_order=2)

generator = torch.Generator(device="cuda")

pipe = DiffusionPipeline.from_pretrained(
  model,
  variant="fp16",
  torch_dtype=torch.float16,
  requires_safety_checker = False,
  use_safetensors=True,
  scheduler=dpm
).to('cuda')

pipe.enable_model_cpu_offload()

pipe.enable_xformers_memory_efficient_attention()

pipe.load_textual_inversion("sd-concepts-library/midjourney-style")

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

prompt_embeds = compel_proc(prompt)

torch.cuda.empty_cache()

pipe.enable_attention_slicing()

image_normal = pipe(
  prompt_embeds=prompt_embeds,
  generator=generator,
  num_inference_steps=20,
  height=1024,
  width=1024,
  guidance_scale=7.5,
  strength=0.6
  ).images[0]

image_saved = save_image(folder_to_save, image_normal, 'normal')

torch.cuda.empty_cache()

init_image = load_image(image_saved)

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline.enable_model_cpu_offload()

pipeline.enable_xformers_memory_efficient_attention()

image_upscaled = pipeline(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale =1.0, height=768, width=768, image=init_image).images[0]

save_image(folder_to_save, image_upscaled, 'upscaled')