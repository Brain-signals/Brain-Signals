import streamlit as st
import requests
from vape_model.utils import show_nii_3D
from vape_model.files import NII_to_3Darray

st.set_page_config(
    page_title="BrainSignal public API", page_icon="🧠", initial_sidebar_state="expanded"
)

st.title('Welcome to BrainSignal public API')

with st.expander("What is BrainSignal ?"):
    """
    BrainSignal is a project carried by Elise Liu, Vincent Attard,
    Pierre Billon and Adrien Combes from le wagon's Data Science batch#862.
    The goal is to use deep learning to deep learning to
    detect Brain Pathologies from 3D MRI scans.
    We created a 3D convolutional neuronal networks ... bla bla bla.
    """

st.header('Try our API')

API_URL = 'https://vape-mri-image-rxbyapf3mq-ew.a.run.app/predict'

volume_file = st.file_uploader(label='Drop your Nifti file',type='nii')

use_example_file = st.checkbox(
    'Or use example file', False, help='Use example file to demo the app'
)

if use_example_file:
    volume_file = open('app/sub-OAS30281_ses-d0042_T1w.nii','rb')

if volume_file is not None:
    bt1 = st.button('Show brain in 3D')
    bt2 = st.button('Show brain in 2D')
    bt3 = st.button('Make Prediction')

    if bt1:
        vol = NII_to_3Darray('app/sub-OAS30281_ses-d0042_T1w.nii')
        st.write(show_nii_3D(vol))

    if bt2:
        vol = NII_to_3Darray('app/sub-OAS30281_ses-d0042_T1w.nii')
        st.write(show_nii_3D(vol))

    if bt3:
        with st.spinner('Processing your Nifti file, please wait...'):
            response = requests.post(API_URL, files={'nii_file': volume_file}).json()
            print(response)
        st.success(f'your prediction is : {response["pred"]}')
