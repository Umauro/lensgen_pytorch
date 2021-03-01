import argparse
import datetime
import os
import errno
import uuid
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from skimage.io import imsave, imshow

#torch stuff
import torch

from src.dcgan_models import Generator
from src.utils.generate_utils import save_image

#Argument Parser

parser = argparse.ArgumentParser(
    description='Arguments for generating images'
)

parser.add_argument(
    '--model_path',
    type=str,
    required=True,
    help='path for trained generator'
)

parser.add_argument(
    '--output_path',
    type=str,
    required=True,
    help='path for output images'
)

parser.add_argument(
    '--num_images',
    type=int,
    required=True,
    help='number of images to generate'
)

# Parse Arguments
args = parser.parse_args()

MODEL_PATH = args.model_path
OUTPUT_PATH = args.output_path
NUM_IMAGES = args.num_images

#create output folder

file_types = ['png','npz']

for file_type in file_types: 
    try:
        os.makedirs(
            '{}/{}'.format(
                OUTPUT_PATH,
                file_type
            )
        )
        print('Output {} folder created'.format(file_type))
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

# TRAIN DATASET STATS
TRAIN_MIN = -4.2955635e-12
TRAIN_MAX = 2.1745163e-09

#load generator model
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
g_model = Generator()
try:
    g_model.load_state_dict(torch.load(MODEL_PATH))
    g_model.to(device)
except FileNotFoundError as error:
    print('Model not found')
except Exception as error:
    raise

#Generate images
BATCH_SIZE = 64
batch_steps = (NUM_IMAGES // BATCH_SIZE) + 1 

for batch_step in tqdm(range(batch_steps)):
    # Determinate the number of images to generate in each step
    if BATCH_SIZE * (batch_step + 1) < NUM_IMAGES:
        n_images = BATCH_SIZE
    else:
        n_images = NUM_IMAGES - BATCH_SIZE * (batch_step)
    noise = torch.randn(n_images,100,device=device)
    with torch.no_grad():
        generated_lens = g_model(noise).detach().cpu()
    generated_lens = np.transpose(
        generated_lens,
        (0,2,3,1)
    ) #Channel last
    
    # Save png and npz
    for lens_index in range(generated_lens.shape[0]):
        image_id = uuid.uuid4()
        lens = generated_lens[lens_index].numpy()
        lens = lens * 0.5 + 0.5 #[0,1] range
        
        save_image(
            lens[:,:,0],
            '{}/png/'.format(OUTPUT_PATH),
            image_id
        )
        
        # lens for npz
        lens = (TRAIN_MAX - TRAIN_MIN)*lens + TRAIN_MIN #denormalize image


        np.savez_compressed(
            '{}/{}/{}.npz'.format(
                OUTPUT_PATH,
                'npz',
                image_id
            ),
            lens
        )
        
        
