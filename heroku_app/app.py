import streamlit as st

import numpy as np
from helpers import *
from preprocess import *
from PIL import Image
import time
import base64


def display_image(volume):
    columns = st.columns(3)

    slice_x = int(volume.shape[0]/2)
    slice_y = int(volume.shape[1]/2)
    slice_z = int(volume.shape[2]/2)

    columns[0].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;"> X axis </p></div>',unsafe_allow_html=True)
    columns[0].image(np.rot90(volume[slice_x,:,:]),width=200)

    columns[1].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;">Y axis</p></div>',unsafe_allow_html=True)
    columns[1].image(np.rot90(volume[:,slice_y,:]),width=200)

    columns[2].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;">Z axis</p></div>',unsafe_allow_html=True)
    columns[2].image(np.rot90(volume[:,:,slice_z]),width=200)

def display_volume(volume):
    st.plotly_chart(plot3D(volume),
                use_container_width=True,
                sharing="streamlit")

#  ------ FRONT-END STARTS HERE ------

with st.sidebar:
    title='<p style="font-family:Courier; color:Blue; font-size: 18px;"> MRI info and params </p>'
    st.markdown(title,unsafe_allow_html=True)


original_title = '<p style="font-family:Courier; color:Blue; font-size: 35px;">Brain Signal</p>'
st.markdown(original_title, unsafe_allow_html=True)
image = Image.open('BrainSignal.png')
st.image(image)

#st.subheader("Brain MRI visualization and analysis")
#image = Image.open('images/wagon.png')
#st.image(image, caption='Le Wagon', use_column_width=False)

upload_mess='<p style="font-family:Courier; color:Black; font-size: 16px;">Please upload your .nii 🧐</p>'
st.markdown(upload_mess,unsafe_allow_html=True)
volume_file = st.file_uploader(' ')

if volume_file is not None:
    with open(volume_file.name, 'wb') as f:
        f.write(volume_file.getbuffer())

    volume=load_volume(volume_file.name)

    # "with" notation
    with st.sidebar:
        volume_initial =volume.shape
        shape_mess=f"""
                <style>
                p.a {{
                font-family: Arial;
                color:Black;
                font-size: 18px;
                }}
                </style>
                <ul>
                <li>Volume dimensions before preprocessing: {volume_initial}</li>
                </ul>
                """
        st.markdown(shape_mess,unsafe_allow_html=True)

    # display original volume shape
    mn = np.min(volume)
    mx = np.max(volume)
    #volume = ((volume/mx) * 255).astype(np.uint8)
    # preprocessing
    #with st.progress('Preprocessing in progress ...'):
        #volume = crop_volume(volume)
        #volume = resize_and_pad(volume)
        #volume= normalize_vol(volume)

    inprog_mes='<p style="font-family:Courier; color:Blue; font-size: 20px;">Preprocessing ... 🏊🏻‍♂️</p>'
    st.markdown(inprog_mes,unsafe_allow_html=True)

    #display the cat dif
    file_ = open("/Users/lison/code/Elise-L/VAPE-MRI/heroku-app/chat attend.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True)

    my_bar = st.progress(0)
    volume = crop_volume(volume)
    volume = resize_and_pad(volume)
    volume= normalize_vol(volume)

    st.write()
    #progression bar
    for percent_complete in range(100):
        time.sleep(0.2)
        my_bar.progress(percent_complete + 1)

    st.balloons()

    with st.sidebar:
        volume_afterpreproc =volume.shape
        shape_mess_2=f"""
                <style>
                p.a {{
                font-family: Arial;
                color:Black;
                font-size: 18px;
                }}
                </style>
                <ul>
                <li>Volume dimensions after preprocessing: {volume_afterpreproc}</li>
                </ul>
                """
        st.markdown(shape_mess_2,unsafe_allow_html=True)

    #st.info(f"volume dimensions after preprocessing: {volume.shape}")

    Image_disp='<p style="font-family:Courier; color:Blue; font-size: 18px;">Processed Images </p>'
    st.markdown(Image_disp,unsafe_allow_html=True)
    st.write()
    # display 3 slices of the .nii files
    display_image(volume)

    #display volume
    display_volume(volume)
