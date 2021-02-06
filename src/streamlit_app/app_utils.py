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

TRAIN_MAX = -4.2955635e-12
TRAIN_MIN = 2.1745163e-09
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

def plot_generated_lens(model,device,n_images):
    noise = torch.randn(n_images,100,device=device)
    with torch.no_grad():
        generated_lens = model(noise).detach().cpu()
    generated_lens = np.transpose(generated_lens,(0,2,3,1))
    fig,axes = plt.subplots(1,n_images)
    plt.subplots_adjust(wspace=0.05)
    for index in range(n_images):
        axes[index].imshow(generated_lens[index,:,:,0],cmap='hot')
        axes[index].axis('off')
    st.pyplot(fig)

def plot_training_losses(model_name):
    csv_path = '{}/{}.csv'.format(RESULTS_PATH,model_name)
    try:
        df = pd.read_csv(csv_path,sep=';')
        train_steps = [x for x in range(len(df))]
        fig,axes = plt.subplots(2,1,figsize=(15,12))
        for index, column_name in enumerate(df.columns):
            axes[index].plot(train_steps,df[column_name],label=column_name)
            axes[index].set_xlabel('Train step',size=15)
            axes[index].set_ylabel('Loss function',size=15)
            axes[index].set_title('{} versus train step'.format(column_name),size=15)
            axes[index].legend()
        st.pyplot(fig)

        
    except OSError as error:
        if error.errno == errno.ENOENT:
            st.text('No se encuentra el archivo {}'.format(csv_path))
        else:
            st.text(error)
