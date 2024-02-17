import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

import base64, os
from IPython.display import HTML
from google.colab.output import eval_js
from base64 import b64decode
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile, rmtree
import shutil
from google.colab import files


from ip_adapter import IPAdapter

base_model_path = "/content/beautifulRealistic_v7"
vae_model_path = "stabilityai/sd-vae-ft-mse-original"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"
images_dir = "/content/images"

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

noise_scheduler = DDIMScheduler(
      num_train_timesteps=1000,
      beta_start=0.00085,
      beta_end=0.012,
      beta_schedule="scaled_linear",
      clip_sample=False,
      set_alpha_to_one=False,
      steps_offset=1,
  )

def image_grid(imgs, rows, cols):
    from PIL import Image
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def using_ipadapter(noise_scheduler, vae, image_path):
  from PIL import Image
  ## Image Variations
  # load SD pipeline
  pipe = StableDiffusionPipeline.from_pretrained(
      base_model_path,
      torch_dtype=torch.float16,
      scheduler=noise_scheduler,
      vae=vae,
      feature_extractor=None,
      safety_checker=None
  )

  image = Image.open(image_path)
  ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
  return ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42)

def create_optical_illusion(image_path1, image_path2):
  torch.cuda.empty_cache()
  pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
      base_model_path,
      torch_dtype=torch.float16,
      scheduler=noise_scheduler,
      vae=vae,
      feature_extractor=None,
      safety_checker=None
  )
  image = Image.open(image_path1)
  g_image = Image.open(image_path2)
  ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
  return ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=0.6)

def create_like_deepfakes(face_path, original_path, masked_path):
  image = Image.open(face_path)
  image.resize((256, 256))
  masked_image = Image.open(original_path).resize((512, 768))
  mask = Image.open(masked_path).resize((512, 768))

  pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
      base_model_path,
      torch_dtype=torch.float16,
      scheduler=noise_scheduler,
      vae=vae,
      feature_extractor=None,
      safety_checker=None
  )

  ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
  return ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
                            seed=42, image=masked_image, mask_image=mask, strength=0.7, )

def download_files(images):
    import zipfile
    import os
    from google.colab import files
    from datetime import datetime

    image_paths = []
    for index, img in enumerate(images):
        save_path = f"{images_dir}/image{(index+1):02d}.png"
        img.save(save_path)
        image_paths.append(save_path)

    # 現在の日付と時刻を 'YYYYMMDDHHMM' の形式で取得
    current_time = datetime.now().strftime('%Y%m%d%H%M')
    
    # ZIPファイル名を 'ip_adapter_YYYYMMDDHHMM.zip' の形式で生成
    zip_filename = f'ip_adapter_{current_time}.zip'

    # もし既存のZIPファイルがあれば削除
    if os.path.exists(zip_filename):
        os.remove(zip_filename)

    # すべての画像ファイルを新しいZIPファイルにまとめます。
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_path in image_paths:
            # ZIPアーカイブに画像ファイルを追加
            zipf.write(img_path)
            # 元の画像ファイルを削除
            os.remove(img_path)

    # ZIPファイルをダウンロードします。
    files.download(zip_filename)
canvas_html = """
<style>
.button {
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
}
</style>
<canvas1 width=%d height=%d>
</canvas1>
<canvas width=%d height=%d>
</canvas>

<button class="button">Finish</button>
<script>
var canvas = document.querySelector('canvas')
var ctx = canvas.getContext('2d')

var canvas1 = document.querySelector('canvas1')
var ctx1 = canvas.getContext('2d')


ctx.strokeStyle = 'red';
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.lineWidth = 5;

var img = new Image();
img.src = "data:image/%s;charset=utf-8;base64,%s";
console.log(img)
img.onload = function() {
  ctx1.drawImage(img, 0, 0);
};
img.crossOrigin = 'Anonymous';

ctx.clearRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = %d
var button = document.querySelector('button')
var mouse = {x: 0, y: 0}

canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.offsetLeft
  mouse.y = e.pageY - this.offsetTop
})
canvas.onmousedown = ()=>{
  ctx.beginPath()
  ctx.moveTo(mouse.x, mouse.y)
  canvas.addEventListener('mousemove', onPaint)
}
canvas.onmouseup = ()=>{
  canvas.removeEventListener('mousemove', onPaint)
}
var onPaint = ()=>{
  ctx.lineTo(mouse.x, mouse.y)
  ctx.stroke()
}

var data = new Promise(resolve=>{
  button.onclick = ()=>{
    resolve(canvas.toDataURL('image/png'))
  }
})
</script>
"""

def draw(imgm, filename='drawing.png', w=400, h=200, line_width=1):
  display(HTML(canvas_html % (w, h, w,h, filename.split('.')[-1], imgm, line_width)))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)

def upload_files():
  from google.colab import files
  uploaded_files = files.upload()
  fnames = list(uploaded_files.keys())

  rmtree(images_dir, ignore_errors=True)
  os.makedirs(images_dir, exist_ok=True)

  current_directory = os.getcwd()

  result = []
  for fn in fnames:
    dest_path = os.path.join(images_dir, fn)
    src_path = os.path.join(current_directory, fn)
    copyfile(src_path, dest_path)
    if os.path.exists(src_path):
      os.remove(src_path)
    result.append(dest_path)
  
  return result

def find_target_file(file_path_list, target_filename="target.png"):
    # os.path.basenameを使用してファイル名のみを取得し、比較する
    target_file_path = next((fp for fp in file_path_list if os.path.basename(fp) == target_filename), None)

    if not target_file_path:
        # ファイルが見つからない場合はメッセージを表示し、Noneを返す
        print(f"{target_filename}がアップロードされていません。")
        return None
    
    # ファイルが見つかった場合はそのパスを返す
    return target_file_path


def pattern01():
  file_path_list = upload_files()
  target_file_path = find_target_file(file_path_list, "face.png")
  if not target_file_path:
    return

  images = using_ipadapter(noise_scheduler, vae, target_file_path)
  download_files(images)

def pattern02():
  file_path_list = upload_files()
  background_file_path = find_target_file(file_path_list, "background.png")
  if not background_file_path:
    return
  target_file_path = find_target_file(file_path_list, "target.jpg")
  if not target_file_path:
    return

  images = create_optical_illusion(background_file_path, target_file_path)
  download_files(images)

def pattern03():
  file_path_list = upload_files()
  target_file_path = find_target_file(file_path_list, "target.png")
  if not target_file_path:
    return
  face_file_path = find_target_file(file_path_list, "target.png")
  if not face_file_path:
    return

  fname = target_file_path
  image64 = base64.b64encode(open(fname, 'rb').read())
  image64 = image64.decode('utf-8')

  print(f'Will use {fname} for inpainting')
  img = np.array(plt.imread(f'{fname}')[:,:,:3])

  target_mask_file_path =f"{images_dir}/target_mask.png" 
  draw(image64, filename=target_mask_file_path, w=img.shape[1], h=img.shape[0], line_width=0.04*img.shape[1])

  with_mask = np.array(plt.imread(f"{images_dir}/target_mask.png")[:,:,:3])
  mask = (with_mask[:,:,0]==1)*(with_mask[:,:,1]==0)*(with_mask[:,:,2]==0)
  plt.imsave(target_mask_file_path, mask, cmap='gray')

  images = create_like_deepfakes(face_file_path, target_file_path, target_mask_file_path)
  download_files(images)