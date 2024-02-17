import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, AutoencoderKL, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from datetime import datetime
from transformers import CLIPImageProcessor, CLIPModel
from PIL import ImageTk, Image
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

## Configuration Variables
camera_definition = ', photo realistic, RAW photo, TanvirTamim, high quality, highres, sharp focus, extremely detailed, cinematic lighting, 8k uhd'

#prompt='1girl, medieval village, nobility, strapless purple dress, bare shoulders, (black hair)1.3, hair up, messy bun, pendant necklace, castle balcony, (masterpiece:1.2), soft lighting, subsurface scattering, heavy shadow, (best quality:1.4), golden ratio, (intricate, high detail:1.2), soft focus'
#prompt='impossibly beautiful portrait of alien shapeshifter entity, insane smile, intricate complexity, surreal horror, inverted neon rainbow drip paint, trending on art station, photoreal, 8 k, octane render by greg rutkowski'
#prompt='(no human:1.9),(nude:1.7) (story board, sketch:1.5), (beautiful female nude:1.3), elegant hipline, with(black eyes)1.6, perfect small breasts with detailed hard nipples, (perky breasts:1.3), beautiful small ass, long messy hair, freckles on nose. captivating eyes, beautiful mouth, perfect face. (dynamic pose:1.5), walking in bamboo forest. highest quality detailed digital painting, ultra detailed. (monochrome:1.4) masterwork composition. side rear view sensual, decadent, erotic'
#prompt = "masterpiece rendering of (leia from star wars:1.5) NSFW slender 20yo. detailed perfect breasts, (visible tiny breasts)+++,  black hair in (side buns:1.3), cute, captivating eyes, beautiful kissable mouth.freckles on nose and cheeks, parted mouth, dynamic pose. (long see through white gown:1.3). (nipples:1.2). cameltoe, (slender:1.3), long elegant hipline. tight flat belly. - (perfect hands:1.4), perfect feet. (interior of death star torture chamber background.:1.1) sci fi. (kneeling on all fours:1.5) unhappy, scared (battered, bloody:1.5) detailed, photographic. unreal engine rendering, ultra quality, 8k. trending on artstation (rear side view)"
#prompt='a beautiful goddess, (fine seethrough silk scarves:1.1), magic glowing shadow, surreal art deco fantasy image. long legs, long slender calves, insanely long messy hair, wearing golden jewelry. small breasts, realistic puffy nipples, perfectly shaped breasts. perky breasts, tight flat belly, small ass, slender thighs. perfect hands, perfect feet. thigh gap, cameltoe.(dynamic pose), ecstatic, sumptuous, decadent dramatic lighting rim light (artwork in the style of Alphonse Mucha, Donato Giancola, Lord Leighton) (rear view:1.9)' + camera_definition
#prompt='glamour portrait shot (from side 60%:0.8), of rich sophisticated female young lady, wrinkles of age, (perfect smile)+++, photorealistic, moody colors, gritty, masterpiece, best quality, (intricate details), (****), eldritch, glow, glowing eyes, (volumetric lighting), unique pose, dynamic pose, dutch angle, 35mm, anamorphic, lightroom, cinematography, film grain, HDR10, 8k hdr, Steve McCurry, ((cinematic)), RAW, color graded portra 400 film, remarkable color, raytracing, subsurface scattering, hyperrealistic, extreme skin details, skin pores, deep shadows, contrast, dark theme,' + camera_definition
#prompt = "photo, 1girl, solo, long hair, breasts, looking at viewer, smile, black hair, navel, jewelry, medium breasts, nipples, standing, nude, outdoors, teeth, day, water, mole, grin, bracelet, completely nude, ring, own hands together, (Sitting with hands clasping in the knee)+++" + camera_definition
#prompt = "abstract beauty, centered, looking at the camera, with black deep eyes, approaching perfection, dynamic, moonlight, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin"
#prompt='full color painting of nude squatting cyberpunk woman with cyberpunk-sunglasses, high heel shoes, (perfect hourglass figure),(perfect perky tits), (((Grafitti art) (by Carne Griffiths))), on a photographic red-brick wall background, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded, blue jeans'
#prompt='portrait half body female Russian concubine with slim fit body, like miranda cohen painting by yoshitaka amano, tsutomu nihei, donato giancola, tim hildebrandt, oil on canvas, trending on artstation, featured on pixiv, cinematic composition, extreme detail, metahuman creator'
#prompt='complex 3d render ultra detailed of a beautiful porcelain profile woman android face (black eyes)1.8, cyborg, robotic parts, 150 mm, beautiful studio soft light, rim light, vibrant details, luxurious cyberpunk, lace, hyperrealistic, anatomical, facial muscles, cable electric wires, microchip, elegant, beautiful background, octane render, H. R. Giger style, 8k, best quality, masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,wallpaper, (realistic, photo-realistic:1.37),Amazing, finely detail, masterpiece,best quality,official art, extremely detailed CG unity 8k wallpaper, absurdres, incredibly absurdres, robot, silver halmet, full body, sitting'
#prompt='a flaming playing card with a spade, in the style of ingrid baars, digital art techniques, alessio albi, caras ionut, dark gray and gold, wallpaper, 8k 3d'
#negative_prompt = '(worst quality:2),(low quality:1.4),(logo,mark:2),(manicure:1.2),(long neck:2)'
#prompt='(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl), extreme detailed,(fractal art:1.3),colorful,highest detailed, (black eyes:1.2)'
#prompt='burdeos dog black and silver abstract beauty, centered, looking at the camera, approaching perfection, dynamic, moonlight, highly detailed, digital painting, artstation, venom costume, concept art, smooth, sharp focus, illustration, art by Carne Griffiths and Wadim Kashin, (very detailed), (phone wallpaper:2, Wallpaper, 8K, very detailed) black eyes, black hair, greyscale, guts \(berserk\), male mole, monochrome, solo, sword, weapon, ((8K wallpaper, super detailed, highres:2)), high quality, photorealistic, unreal engine, high res'
#prompt='a photorealistic, 1women, venezuelan woman, with dark skin, black eyes, and black hair, beauty, fit, like miranda cohen, (1girl), extreme detailed, colorful,highest detailed, detailed symmetric realistic face, cinematic composition medium body photo'
#prompt = "a red cat playing with a ball++"
#prompt = "(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2) a door, with a checkered floor. and two columns on each side, the ceiling is the sky, and as you pass the door you see an angel, female, with beautiful wings. blond hair."
#prompt='(photo of Filipina women:1),Pinays,a Pinay with light brown skin with glasses and hair highlights, small breasts AND a Pinay with brown skin,wristwatch AND a Pinay with pale skin with long black hair,BREAK photo of Filipina women flashing breasts,nude,(photo of a group of Filipina women completely nude:1.5),standing,looking at viewer,group selfie,instagram filter,epiCPhoto,blurry,thong panty,undressing panty,small breasts,large breasts,open mouth,undressing,topless,wristwatch,pale skin,pubic hair,classroom,1girl sitting,legs spread,pussy,1girl standing'
#prompt='The picture is a selfie of a woman in a home studio setting. She wears large, black-framed glasses and over-ear headphones, suggesting she may be engaged in audio work, like podcasting, music production, or streaming. Her hair is styled in two casual braids, and she wears a sleeveless top. The room is illuminated by soft lighting, and behind her is a window with blinds, a microphone with a pop filter, and various other electronic equipment. The setting appears to be a personal workspace, neatly organized, with a homey ambiance.'
prompt = "photo, 1girl, solo, long hair, breasts, looking at viewer, smile, black hair, navel, jewelry, medium breasts, nipples, standing, nude, outdoors, teeth, day, water, mole, grin, bracelet, completely nude, ring, own hands together, (Sitting with hands clasping in the knee)+++" + camera_definition
#negative_prompt='(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)'
negative_propmt='(disfigured:1.3), (bad art:1.3), (deformed:1.3),(extra limbs:1.3),(close up:1.3),(b&w:1.3), weird colors, blurry, (duplicate:1.5), (morbid:1.3), (mutilated:1.3), [out of frame], extra fingers, mutated hands, (poorly drawn hands:1.3), (poorly drawn face:1.3), (mutation:1.5), (deformed:1.5), (ugly:1.3), blurry, (bad anatomy:1.3), (bad proportions:1.5), (extra limbs:1.3), cloned face, (disfigured:1.5), out of frame,  (malformed limbs:1.1), (missing arms:1.3), (missing legs:1.3), (extra arms:1.5), (extra legs:1.5), mutated hands, (fused fingers:1.1), (too many fingers:1.1), (long neck:1.5), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy'
vae_model_path = "../2-models/vae/vae-ft-mse-84-ema-pruned"
model="../2-models/difussers/juggernautXL_"

## in positive: solid-best-physical-appearance-perfect::2 in negative: not-solid-best-physical-appearance-perfect::2

#dpm = DPMSolverMultistepScheduler.from_pretrained('model', subfolder="scheduler", solver_order=2)

generator = torch.Generator(device="cuda")

pipe = DiffusionPipeline.from_pretrained(
  model,
  variant="fp16",
  torch_dtype=torch.float16,
  use_safetensors=True,
).to('cuda')

pipe.enable_xformers_memory_efficient_attention()
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")

compel = Compel(
  tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
  text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
  requires_pooled=[False, True]
)

conditioning, pooled = compel(prompt)

torch.cuda.empty_cache()

pipe.enable_attention_slicing()

image = pipe(
  prompt_embeds=conditioning,
  pooled_prompt_embeds=pooled,
  generator=generator,
  num_inference_steps=20,
  height=1024,
  width=1024,
  guidance_scale=7.5,
  strength=0.6
  ).images[0]

save_image(folder_to_save, image)