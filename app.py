from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
from io import BytesIO
import os
import numpy as np
from PIL import Image
import cv2
import torch
import random

os.system("pip install -e ./controlnet_aux")

from controlnet_aux import CannyDetector

from huggingface_hub import hf_hub_download

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN_GATED")
login(token=hf_token)

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet])
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

canny = CannyDetector()

torch.backends.cuda.matmul.allow_tf32 = True
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload() # for saving memory

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def extract_canny(image):
    processed_image_canny = canny(image)
    return processed_image_canny

def apply_gaussian_blur(image, kernel_size=(21, 21)):
    image = convert_from_image_to_cv2(image)
    blurred_image = convert_from_cv2_to_image(cv2.GaussianBlur(image, kernel_size, 0))
    return blurred_image

def convert_to_grayscale(image):
    image = convert_from_image_to_cv2(image)
    gray_image = convert_from_cv2_to_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return gray_image

def add_gaussian_noise(image, mean=0, sigma=10):
    image = convert_from_image_to_cv2(image)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = convert_from_cv2_to_image(np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return noisy_image

def tile(input_image, resolution=768):
    input_image = convert_from_image_to_cv2(input_image)
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    img = convert_from_cv2_to_image(img)
    return img

def resize_img(input_image, max_side=1024, min_side=768, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/infer', methods=['POST'])
def infer():
    data = request.form
    files = request.files

    prompt = data.get("prompt", "best quality")
    control_strength = float(data.get("controlStrength", 0.7))
    control_guidance_end = float(data.get("controlGuidanceEnd", 0.8))
    seed = int(data.get("seed", 42))
    randomize_seed = data.get("randomizeSeed", "false").lower() == "true"

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    cond_in = files.get("controlImage")
    # image_in = files.get("referenceImage")

    if cond_in:
        control_image = resize_img(Image.open(cond_in))
        control_image = extract_canny(control_image) # added this cuz im only passing a reference img   
    # elif image_in:
    #     image_in = resize_img(Image.open(image_in))
    #     control_image = extract_canny(image_in)
    else:
        return jsonify({"error": "No input image provided"}), 400

    width, height = control_image.size

    image = pipe(
        prompt,
        control_image=[control_image],
        control_guidance_end=[control_guidance_end],
        width=width,
        height=height,
        controlnet_conditioning_scale=[control_strength],
        num_inference_steps=24,  # Default value
        guidance_scale=3.5,      # Default value
        generator=torch.manual_seed(seed),
    ).images[0]

    torch.cuda.empty_cache()

    # Save images to BytesIO for response
    result_io = BytesIO()
    control_io = BytesIO()
    image.save(result_io, format="PNG")
    control_image.save(control_io, format="PNG")
    result_io.seek(0)
    control_io.seek(0)

    return jsonify({
        "result_image": result_io.getvalue().decode("latin1"),
        "control_image": control_io.getvalue().decode("latin1")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)