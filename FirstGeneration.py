import gradio as gr

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
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

def save_image(folder_to_save, image):
  current_time = datetime.now().strftime('%Y%m%d%H%M%S')
  image_path = f"{(folder_to_save)}/px_{(current_time)}_juggernautXL.png"
  image.save(image_path)

def initialize_environment():
  torch.cuda.empty_cache()


initialize_environment()
folder_to_save = generate_folders()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipeline = StableDiffusionPipeline.from_single_file("../2-models/raw/toonBabes_v10.safetensors", revision="fp16",
  torch_dtype=torch.float16,
  safety_checker = None,
  requires_safety_checker = False,
)
pipeline.to("cuda")

pipeline.safety_checker = None
pipeline.requires_safety_checker = False

images = pipeline("(masterpiece,  best quality,  dynamic angle),  surrealistic,  colorful,  (nsfw:1.2), Thomasin McKenzie,  fishnet,  long sleeves,  chain necklace,  thigh straps,  bracelet,  sneakers,  jewelery,  gold,  yellow hair,  ponytail,  frowning,  makeup,  fit,  abs,  large breasts,  nipples,  pussy,  anus, sitting on the ground,  arms up, teal,  wall background, (detailed face,  detailed eyes),  looking away, God rays,  hard shadow,  dark studio,  rim lighting,  dimly lit,  low key,  mrrpss, <lora:EMS-179-EMS:0.600000>, , <lora:EMS-69762-EMS:0.500000>, , <lora:EMS-75851-EMS:-1.600000>, , <lora:EMS-8103-EMS:0.600000>, , <lora:EMS-75843-EMS:-0.600000>, , <lora:EMS-75835-EMS:1.400000>", negative_prompt = 'ng_deepnegative_v1_75t, badhandv4, Asian-Less-Neg, (loli), young, thicc, chubby, fat, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, multiple limbs, extra limbs, deformed limbs, normal quality, ((monochrome)), ((grayscale)), cum, wet pussy')

print(images)

with autocast('cuda'):
  image = images[0][0]

image_path = "/home/dav1nx1/Code/1-dav1nx1/1-LLM-HUGGINFACE/1-tutorial/generated.png"

save_image(folder_to_save, image)
# import diffusers
# import transformers

# import sys
# import os
# import shutil
# import time

# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# from PIL import Image

# if torch.cuda.is_available():
#     device_name = torch.device("cuda")
#     torch_dtype = torch.float16
# else:
#     device_name = torch.device("cpu")
#     torch_dtype = torch.float32