import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.streamlit_app.app_utils import (
    plot_training_losses,
    get_available_models,
    plot_generated_lens, 
    load_model
)

#====================#
#      Side bar      #
#====================#
models_list = get_available_models()
model_name = st.sidebar.selectbox(
    'Select generator model',
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
    st.header('Example generated lens')
    plot_generated_lens(g_model,device)

    st.header('Training losses')
    plot_training_losses(model_name)
else:
    st.warning('No generator models available')