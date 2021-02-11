import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.streamlit_app.app_utils import (
    plot_training_losses,
    get_available_models,
    plot_generated_lens, 
    load_model,
    plot_interpolation,
    train_help,
    sidebar_intro,
    show_training_progress_gif
)

st.set_page_config(
    page_title="LensGen results",
    page_icon="static/app_icon.png"
)

#====================#
#      Side bar      #
#====================#

sidebar_intro()
st.sidebar.header('Select experiment model')
models_list = get_available_models()
model_name = st.sidebar.selectbox(
    'Generator model',
    models_list
)
if len(models_list):
    g_model, device = load_model(model_name)
    show_content = True
else:
    show_content = False
    

#====================#
#   Principal page   #
#====================#
st.title('Lensgen train results')
if show_content:
    # Example images
    st.header('Example generated lens')
    n_images = st.slider('Select the number of images',min_value=1,max_value=10,value=5)
    plot_generated_lens(g_model,device,n_images)

    # Training stuff
    st.header('Training losses')
    plot_training_losses(model_name)

    with st.beta_expander("Training progress GIF"):
        show_training_progress_gif(model_name)
    
    # Latent space
    st.header('Latent space interpolation')
    n_interpolations = st.slider(
        'Select the number of interpolations experiments',
        min_value=2,
        max_value=10,
        value=5
    )
    n_points = st.slider(
        'Select the number of interpolated points for each experiment',
        min_value=2,
        max_value=10,
        value=5
    )
    plot_interpolation(g_model,device,n_interpolations,n_points)

else:
    st.warning('No generator models available')
    train_help()
