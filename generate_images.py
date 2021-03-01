import argparse
import os
import errno
import numpy as np

from tqdm.auto import tqdm

#torch stuff
import torch

from src.dcgan_models import Generator

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
timestamp = datetime.datetime.now().strftime("%d-%m-%y-%H%M%S")

for file_type in file_types: 
    try:
        os.makedirs(
            '{}/{}/{}'.format(
                OUTPUT_PATH,
                timestamp,
                file_type
            )
        )
        print('Output folder created')
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

# TRAIN DATASET STATS
TRAIN_MIN = -4.2955635e-12
TRAIN_MAX = 2.1745163e-09

