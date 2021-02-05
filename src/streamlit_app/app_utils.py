import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import errno
import glob
import torch 

from torchvision import utils as vutils
from src.dcgan_models import Generator


plt.style.use('seaborn') #plot styles

MODELS_PATH = 'models'
RESULTS_PATH = 'results/train'

def get_available_models():
    models_list = glob.glob('{}/*.pt'.format(MODELS_PATH))
    models_list = [x.split('\\')[1].split('.')[0] for x in models_list]
    return models_list

@st.cache
def load_model(model_name):
    # Set Device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    g_model = Generator()
    g_model.load_state_dict(torch.load('{}/{}.pt'.format(MODELS_PATH,model_name)))
    g_model.to(device)
    return g_model, device

def plot_generated_lens(model,device):
    noise = torch.randn(5,100,device=device)
    with torch.no_grad():
        generated_lens = model(noise).detach().cpu()
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(np.transpose(vutils.make_grid(generated_lens,padding=5,normalize=True),(1,2,0)))
    st.pyplot(fig)

def plot_training_losses(model_name):
    csv_path = '{}/{}.csv'.format(RESULTS_PATH,model_name)
    try:
        df = pd.read_csv(csv_path,sep=';')
        train_steps = [x for x in range(len(df))]
        fig = plt.figure(figsize=(15,7))
        plt.plot(train_steps,df['d_loss'],label='Discriminator Loss')
        plt.plot(train_steps,df['g_loss'],label='Generator Loss')
        plt.xlabel('Train step',fontsize=20)
        plt.ylabel('Loss function',fontsize=20)
        plt.legend(
            fontsize=20,
            loc=1,
            edgecolor='inherit',
            frameon=True,
            shadow=True
        )
        st.pyplot(fig)

        
    except OSError as error:
        if error.errno == errno.ENOENT:
            st.text('No se encuentra el archivo {}'.format(csv_path))
        else:
            st.text(error)
