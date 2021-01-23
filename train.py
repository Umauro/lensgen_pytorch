import argparse
import os
import errno
from tqdm.auto import tqdm

#torch stuff
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

# utils and networks models
from utils.dataset import SpaceBasedDataset, PreprocessLensImage
from utils.train_utils import label_noise
from dcgan_models import Discriminator, Generator, weights_init


parser = argparse.ArgumentParser(
    description='Arguments for train process'
)

parser.add_argument(
    'csv_path',
    type=str,
    help='path for csv file with image annotations'
)

parser.add_argument(
    'image_folder_path',
    type=str,
    help='path for images folder'
)

parser.add_argument(
    'epochs',
    type=int,
    help='number of epochs for training'
)

parser.add_argument(
    'batch_size',
    type=int,
    help='size of minibatch for training'
)

parser.add_argument(
    'generator_save_path',
    type=str,
    help='path for final generator weights'
)

# PARSE ARGUMENTS
args = parser.parse_args()

CSV_PATH = args.csv_path
IMAGE_FOLDER_PATH = args.image_folder_path
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
GENERATOR_SAVE_PATH = args.generator_save_path

# Create save folder if not exists
try:
    os.mkdir(GENERATOR_SAVE_PATH)
    print('Generator folder created')
except OSError as error:
    if error.errno != errno.EEXIST:
        raise

# TRAIN DATASET STATS
TRAIN_MAX = -4.2955635e-12
TRAIN_MIN = 2.1745163e-09

#Instanciate Dataset and Dataloader
lens_dataset = SpaceBasedDataset(
    CSV_PATH,
    IMAGE_FOLDER_PATH,
    transform=transforms.Compose(
        [
            PreprocessLensImage(TRAIN_MAX,TRAIN_MIN)
        ]
    )
)

dataloader = DataLoader(
    lens_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# Set Device
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# Instanciate GAN networks
d_model = Discriminator().to(device)
d_model.apply(weights_init)

g_model = Generator().to(device)
g_model.apply(weights_init)


# Train Setup
LR = 0.00009
BETA_1 = 0.5
BETA_2 = 0.999
FRAC = 0.05

real_label = 0.9 # For One-sided Label Smoothing
synthetic_label = 0

loss_function = nn.BCELoss()

d_optimizer = optim.Adam(d_model.parameters(),lr=LR,betas=(BETA_1,BETA_2))
g_optimizer = optim.Adam(g_model.parameters(),lr=LR,betas=(BETA_1,BETA_2))

#Begin Training Loop
d_losses = []
g_losses = []

for epoch in range(EPOCHS):
    progress_bar = tqdm(dataloader)
    for train_step, real_data in enumerate(progress_bar):
        #======================#
        # Discriminator Train  #
        #======================#
        # Real Batch
        d_model.zero_grad() #set gradients to 0
        real_inputs = real_data.to(device)
        batch_size = real_inputs.size(0)
        labels = label_noise(
            torch.full(
                (batch_size,),
                real_label,
                dtype=torch.float,
                device=device
            ),
            FRAC
        ) #apply noisy labels
        
        real_outputs = d_model(real_inputs).view(-1) #Forward pass
        
        d_loss_real = loss_function(real_outputs,labels) #loss calculation
        d_loss_real.backward() #Backward pass
        
        #Synthetic Batch
        noise = torch.randn(batch_size,100,device=device) #sample from N(0,1)
        synthetic_inputs = g_model(noise)
        labels.fill_(synthetic_label) #set labels to 0
        labels = label_noise(labels,FRAC) #noisy labels c:
        synthetic_outputs = d_model(synthetic_inputs.detach()).view(-1) #Forward pass
        d_loss_synthetic = loss_function(synthetic_outputs,labels) #loss calculation
        d_loss_synthetic.backward() #Backward pass
        d_loss = d_loss_real + d_loss_synthetic
        d_losses.append(d_loss)

        d_optimizer.step() # discriminator train step

        #===================#
        # Generator  Train  #
        #===================#
        g_model.zero_grad()
        noise = torch.randn(batch_size,100,device=device)
        synthetic_inputs = g_model(noise)
        labels.fill_(real_label) #Set labels to 1 to mislead the discriminator
        synthetic_outputs = d_model(synthetic_inputs.detach()).view(-1)
        g_loss = loss_function(synthetic_outputs,labels) # forward pass
        g_loss.backward() # backward pass

        g_losses.append(g_loss)

        g_optimizer.step() # Generator train step
        progress_bar.set_postfix(d_loss = d_loss.item(),g_loss= g_loss.item())

#save generator model
torch.save(
    g_model.state_dict(),
    '{}/generator_{}_epoch.pt'.format(
        GENERATOR_SAVE_PATH,
        EPOCHS
    )
) 
