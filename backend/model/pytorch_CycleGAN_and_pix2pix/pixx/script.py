import os
import sys
# Add the parent directory of script.py to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.networks import define_G
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
import os


def img_to_sketch(original_image):
    base_dir = os.path.abspath(os.path.dirname(__file__))  # This gets the directory where script.py is located
    checkpoint_path = os.path.join(base_dir, 'checkpoints', 'latest_net_G (1).pth')
    model_dict = torch.load(checkpoint_path)
    new_dict = OrderedDict()
    for k, v in model_dict.items():
        # load_state_dict expects keys with prefix 'module.'
        new_dict["module." + k] = v

    # make sure you pass the correct parameters to the define_G method
    generator_model = define_G(input_nc=3,output_nc=3,ngf=64,netG="resnet_9blocks",
                                norm="instance",use_dropout=False,init_gain=0.02,gpu_ids=[])

    cleaned_dict = {k.replace("module.", ""): v for k, v in new_dict.items()}

    generator_model.load_state_dict(cleaned_dict)


    # Define image transformation (matching training/testing settings)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # transformed_image = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(256)])(original_image)
    input_tensor = transform(original_image).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_model.to(device)
    input_tensor = input_tensor.to(device)
    generator_model.eval()
    with torch.no_grad():
        output_tensor = generator_model(input_tensor)
    output_tensor = output_tensor.squeeze().cpu()  # Remove batch dimension
    output_tensor = output_tensor * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
    output_image = transforms.ToPILImage()(output_tensor)

    return output_image






