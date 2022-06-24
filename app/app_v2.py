import streamlit as st
import requests
from vape_model.utils import show_nii_3D, show_nii_2D
from vape_model.files import NII_to_3Darray
from PIL import Image

st.set_page_config(
    page_title="BrainSignal public API",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

st.title('Welcome to BrainSignal public API')

# image = Image.open('./Streamlit_folder/BrainSignal_.png')
# st.image(image)

st.header('Try our API')

API_URL = 'https://vape-mri-image-rxbyapf3mq-ew.a.run.app/predict'

volume_file = st.file_uploader(
    label='👇 Drop your Nifti file here to access other functions',
    type='nii',
    help='Only .nii files are accepted yet')

if volume_file is not None:

    st.caption('Please choose an option :', unsafe_allow_html=False)
    col1, col2, col3 = st.columns(3)

    bt1 = col1.button('Show brain in 3D')
    bt2 = col2.button('Show brain in 2D')
    bt3 = col3.button('Make Prediction')

    if bt1:
        vol = NII_to_3Darray('app/sub-OAS30281_ses-d0042_T1w.nii')
        st.write(show_nii_3D(vol))

    if bt2:
        vol = NII_to_3Darray('app/sub-OAS30281_ses-d0042_T1w.nii')
        st.write(show_nii_2D(vol))

    if bt3:
        with st.spinner('Processing your Nifti file, please wait...'):
            response = requests.post(API_URL, files={'nii_file': volume_file}).json()
            print(response)
        st.success(f'your prediction is : {response["pred"]}')

st.header('About')

with st.expander("What is BrainSignal ?"):
    """
    BrainSignal is a project carried by Elise Liu, Vincent Attard,
    Pierre Billon and Adrien Combes from le wagon's Data Science batch#862.
    The goal is to use deep learning to deep learning to
    detect Brain Pathologies from 3D MRI scans.
    We created a 3D convolutional neuronal networks ... bla bla bla.
    """
