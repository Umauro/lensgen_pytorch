import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from utils.dataset import SpaceBasedDataset, PreprocessLensImage
from dcgan_models import Discriminator, Generator, weights_init
from tqdm.auto import tqdm 

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
# PARSE ARGUMENTS
args = parser.parse_args()

CSV_PATH = args.csv_path
IMAGE_FOLDER_PATH = args.image_folder_path
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

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

real_label = 1
synthetic_label = 0

loss_function = nn.BCELoss()

d_optimizer = optim.Adam(d_model.parameters(),lr=LR,betas=(BETA_1,BETA_2))
g_optimizer = optim.Adam(g_model.parameters(),lr=LR,betas=(BETA_1,BETA_2))

#Begin Training Loop
d_losses = []
g_losses = []

for epoch in range(EPOCHS):
    for train_step, real_data in enumerate(tqdm(dataloader)):
        #======================#
        # Discriminator Train  #
        #======================#
        # Real Batch
        d_model.zero_grad() #set gradients to 0
        real_inputs = real_data.to(device)
        batch_size = real_inputs.size(0)
        labels = torch.full(
            (batch_size,),
            real_label,
            dtype=torch.float,
            device=device
        )
        
        real_outputs = d_model(real_inputs).view(-1) #Forward pass
        
        d_loss_real = loss_function(real_outputs,labels) #loss calculation
        d_loss_real.backward() #Backward pass
        
        #Synthetic Batch
        noise = torch.randn(batch_size,100,device=device) #sample from N(0,1)
        synthetic_inputs = g_model(noise)
        labels.fill_(synthetic_label) #set labels to 0

        synthetic_outputs = d_model(synthetic_inputs.detach()).view(-1) #Forward pass
        d_loss_synthetic = loss_function(synthetic_outputs,labels) #loss calculation
        d_loss_synthetic.backward() #Backward pass

        d_losses.append(d_loss_real + d_loss_synthetic)

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
