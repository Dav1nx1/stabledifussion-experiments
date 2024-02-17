import torch
from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from datetime import datetime
import os

## Configuration Variables
#prompt = "a photorealistic stunningly beautiful woman without make-up, (vista posterior:1.2), extremely detailed blue eyes, detailed symmetric realistic face, natural skin texture, extremely detailed skin with skin pores, peach fuzz, messy hair, wearing shawl over her head, masterpiece, absurdres, nikon d850 film stock photograph, kodak portra 400 camera f1.6 lens, extremely detailed, amazing, fine detail, rich colors, hyper realistic lifelike texture, dramatic lighting, unrealengine, trending on artstation, cinestill 800 tungsten, looking at the viewer, photo realistic, RAW photo, TanvirTamim, high quality, highres, sharp focus, extremely detailed, cinematic lighting, 8k uhd"
#prompt = "abstract beauty, centered, looking at the camera, with black deep eyes, approaching perfection, dynamic, moonlight, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin"
#prompt='full color painting of nude squatting cyberpunk woman with cyberpunk-sunglasses, high heel shoes, (perfect hourglass figure),(perfect perky tits), (((Grafitti art) (by Carne Griffiths))), on a photographic red-brick wall background, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded, blue jeans'
#prompt='portrait half body female Russian concubine with slim fit body, like miranda cohen painting by yoshitaka amano, tsutomu nihei, donato giancola, tim hildebrandt, oil on canvas, trending on artstation, featured on pixiv, cinematic composition, extreme detail, metahuman creator'
#prompt='complex 3d render ultra detailed of a beautiful porcelain profile woman android face, cyborg, robotic parts, 150 mm, beautiful studio soft light, rim light, vibrant details, luxurious cyberpunk, lace, hyperrealistic, anatomical, facial muscles, cable electric wires, microchip, elegant, beautiful background, octane render, H. R. Giger style, 8k, best quality, masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,wallpaper, (realistic, photo-realistic:1.37),Amazing, finely detail, masterpiece,best quality,official art, extremely detailed CG unity 8k wallpaper, absurdres, incredibly absurdres, robot, silver halmet, full body, sitting'
#prompt='a flaming playing card with a spade, in the style of ingrid baars, digital art techniques, alessio albi, caras ionut, dark gray and gold, wallpaper, 8k 3d'
#negative_prompt = '(worst quality:2),(low quality:1.4),(logo,mark:2),(manicure:1.2),(long neck:2)'
#prompt='(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl), extreme detailed,(fractal art:1.3),colorful,highest detailed, (black eyes:1.2)'
#prompt='burdeos dog black and silver abstract beauty, centered, looking at the camera, approaching perfection, dynamic, moonlight, highly detailed, digital painting, artstation, venom costume, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin, (very detailed), (phone wallpaper:2, Wallpaper, 8K, very detailed) black eyes, black hair, greyscale, guts \(berserk\), male mole, monochrome, solo, sword, weapon, ((8K wallpaper, super detailed, highres:2)), high quality, photorealistic, unreal engine, high res'
#prompt='a photorealistic, 1women, venezuelan woman, with dark skin, black eyes, and black hair, beauty, fit, like miranda cohen, (1girl), extreme detailed, colorful,highest detailed, detailed symmetric realistic face, cinematic composition medium body photo'

prompt=prompt='The image showcases an anthropomorphic raccoon with a sci-fi, fantasy theme. It stands upright in a confident pose, holding a glowing blue bottle emitting smoke or vapor, suggesting a potion or magical substance. The raccoon is garbed in a detailed, armored suit with futuristic elements, reminiscent of a space adventurer or rogue hero. Its eyes glow with the same eerie blue as the bottle, enhancing the mystical atmosphere. The background is a fusion of dark purples and blues, hinting at a neon-lit urban environment or a starry night sky, which adds to the otherworldly ambiance of the illustration'

negative_prompt='(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)'
vae_model_path = "../2-models/vae/vae-ft-mse-84-ema-pruned"
model="../2-models/difussers/cyberrealistic_v40"

dpm = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")

generator = torch.Generator(device="cuda").manual_seed(53)

pipe = DiffusionPipeline.from_pretrained(
  model, variant="fp16", torch_dtype=torch.float16, use_safetensors=True
)

pipe.to('cuda')
image = pipe(prompt, num_inference_steps=30, negative_prompt=negative_prompt, height=768, width=512, guidance_scale=6, generator=generator, text_encoder=pipe.text_encoder,).images[0]
current_time = datetime.now().strftime('%Y%m%d%H%M%S')

folder_name = "generated"
folder_to_save = os.path.join(os.getcwd(), folder_name)

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
image_path = f"{(folder_to_save)}/px_{(current_time)}_juggernautXL.png"


image.save(image_path)