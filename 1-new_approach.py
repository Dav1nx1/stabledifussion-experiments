from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_single_file("../2-models/raw/juggernautXL_version6Rundiffusion.safetensors")
pipe.save_pretrained("../2-models/difussers/juggernautXL_")

