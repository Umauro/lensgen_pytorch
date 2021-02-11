import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import errno
import glob
import torch 
import base64

from torchvision import utils as vutils
from sklearn.metrics.pairwise import cosine_similarity
from src.dcgan_models import Generator


plt.style.use('seaborn') #plot styles

LATENT_DIM = 100
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
            st.warning('No se encuentra el archivo {}'.format(csv_path))
        else:
            st.alert(error)


def slerp(p_0,p_1,t):
    '''
        Retorna un punto entre otros dos puntos definidos utilizando
        interpolación linear esférica
        
        Returns a interpolated point between to other points using 
        SLERP.

        Params:
            - p_0: Numpy Array, first point
            - p_1: Numpy Array, second point
            - t: Float between 0 y 1. Ajust the position between p_0 and p_1
    '''
    assert t >= 0 and t <= 1, "t must be between 0 and 1"
    theta = np.arccos(cosine_similarity(p_0,p_1)) # angle between p_0 y p_1
    return (((np.sin(1 - t)*theta)/np.sin(theta)) * p_0) + ((np.sin(t*theta))/(np.sin(theta)))*p_1

def plot_interpolation(model,device,n_interpolations,n_points):
    '''
        Plot a grid of n_interpolations x n_points interpolated images

        params:
            - model: Generator object
            - device: Pytorch device
            - n_interpolations: number of interpolation experiments
            - n_points: number of points to interpolate
    '''
    with st.spinner('Interpolating...'):
        fig, axes = plt.subplots(n_interpolations,n_points)
        plt.subplots_adjust(wspace=0.05,hspace=0.05)
        for row in range(n_interpolations):
            p_0 = np.random.normal(0,1,(1,LATENT_DIM)).astype(np.float32) #random points
            p_1 = np.random.normal(0,1,(1,LATENT_DIM)).astype(np.float32)
            for col, t in enumerate(np.linspace(0,1,n_points)):
                new_point = torch.from_numpy(slerp(p_0,p_1,t)).to(device)
                with torch.no_grad():
                    generated_lens = model(new_point).detach().cpu()
                generated_lens = np.transpose(generated_lens,(0,2,3,1))
                axes[row][col].imshow(generated_lens[0,:,:,0],cmap='hot')
                axes[row][col].axis('off')
        st.pyplot(fig)

def train_help():
    st.markdown(
        """
        You must run the `train.py` script to see some results.

        Params:
        - **csv_path:** path to csv file with dataset annotations.
        - **image_folder_path:** path to FITS images folder.
        - **epochs:** training epochs.
        - **batch_size:** mini batch size.
        - **save_progress:** generate gif image from fixed noise during training **(OPTIONAL)**.


            python train.py --csv_path CSV_PATH --image_folder_path IMAGE_FOLDER_PATH \
            
            --epochs EPOCHS --batch_size BATCH_SIZE [--save_progress | --no_save_progress] 

        #### Example

            python train.py --csv_path data/SpaceBasedTraining/classifications.csv \
            
                            --image_folder_path data/SpaceBasedTraning/Public/Band1 \
                
                            --epochs 100 --batch_size 64 --save_progress
        """
    )

def sidebar_intro():
    st.sidebar.markdown(
        """
        # LensGEN: Generador de imágenes artificiales de lentes gravitacionales vía GAN

        This work is my bachelor's thesis. Originally was made in Keras, but i made 
        a Pytorch version in order to learn this framework.

        The original keras version, thesis, and notebooks are available on this [github repo](https://github.com/Umauro).
        
        Thesis defense is available on [Youtube](https://www.youtube.com/watch?v=n5wXqbQrQdk&ab_channel=DepartamentodeInform%C3%A1ticaUTFSM)
        
        ---
        """   
    )

def show_training_progress_gif(model_name):
    @st.cache
    def generate_gif_url(gif_path):
        with open(gif_path,"rb") as gif_file:
            gif_content = gif_file.read()
            gif_url = base64.b64encode(gif_content).decode("utf-8")
        return gif_url
    gif_path = '{}/{}.gif'.format(RESULTS_PATH,model_name)
    gif_url = generate_gif_url(gif_path)
    st.markdown(
        """
        <img src="data:image/gif;base64,{}" alt="train_progress" style="max-width:100%;height:auto;">
        """.format(gif_url),
        unsafe_allow_html=True
    )