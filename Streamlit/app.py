import streamlit as st
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import time
import cv2
from skimage import io
import plotly.graph_objects as go
import requests
from termios import CR1
from vape_model.preprocess import *
import base64
import io as iioo

API_URL = "http://127.0.0.1:8000/predict"


def display_image(volume):
    columns = st.columns(3)

    slice_x = int(volume.shape[0]/2)
    slice_y = int(volume.shape[1]/2)
    slice_z = int(volume.shape[2]/2)

    columns[0].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;"> X axis </p></div>',unsafe_allow_html=True)
    columns[0].image(np.rot90(volume[slice_x,:,:]),width=300)

    columns[1].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;">Y axis</p></div>',unsafe_allow_html=True)
    columns[1].image(np.rot90(volume[:,slice_y,:]),width=300)

    columns[2].markdown('<div style="text-align: center"><p style="font-family:Courier; color:Black; font-size: 18px;">Z axis</p></div>',unsafe_allow_html=True)
    columns[2].image(np.rot90(volume[:,:,slice_z]),width=300)

def display_volume(volume):
    st.plotly_chart(plot3D(volume),
                use_container_width=True,
                sharing="streamlit")

def plot3D(img):
  r, c = img.shape[0:2]
  img = ndimage.zoom(img,np.array([96/img.shape[0], 96/img.shape[1], 1]))

  # Define frames
  nb_frames = img.shape[-1] - 1 #minus 1 frame for initial display

  fig = go.Figure(frames=[go.Frame(data=go.Surface(z=(nb_frames/10 - k * 0.1) * np.ones((96, 96)),
                                                   surfacecolor=np.rot90(img[:,:,nb_frames - k]),
                                                   cmin=0, cmax=255
                                                   ),
                                   name=str(k) # you need to name the frame for the animation to behave properly
                                   )
                  for k in range(20,nb_frames-10)])

  # Add data to be displayed before animation starts
  fig.add_trace(go.Surface(
      z=nb_frames/10 * np.ones((96, 96)),
      surfacecolor=np.rot90(img[:,:,nb_frames]),
      colorscale='Gray',
      cmin=0, cmax=255,
      colorbar=dict(thickness=20, ticklen=4)
      ))


  def frame_args(duration):
      return {
              "frame": {"duration": duration},
              "mode": "immediate",
              "fromcurrent": True,
              "transition": {"duration": duration, "easing": "linear"},
          }

  sliders = [
              {
                  "pad": {"b": 10, "t": 60},
                  "len": 0.9,
                  "x": 0.1,
                  "y": 0,
                  "steps": [
                      {
                          "args": [[f.name], frame_args(0)],
                          "label": str(k),
                          "method": "animate",
                      }
                      for k, f in enumerate(fig.frames)
                  ],
              }
          ]

  # Layout
  fig.update_layout(
          title='Slices in volumetric data',
          width=600,
          height=600,
          scene=dict(
                      zaxis=dict(range=[0, nb_frames/10], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1),
                      ),
          updatemenus = [
              {
                  "buttons": [
                      {
                          "args": [None, frame_args(50)],
                          "label": "&#9654;", # play symbol
                          "method": "animate",
                      },
                      {
                          "args": [[None], frame_args(0)],
                          "label": "&#9724;", # pause symbol
                          "method": "animate",
                      },
                  ],
                  "direction": "left",
                  "pad": {"r": 10, "t": 70},
                  "type": "buttons",
                  "x": 0.1,
                  "y": 0,
              }
          ],
          sliders=sliders
    )
  return fig

def load_image(image_file):
	img = Image.open(image_file)
	return img

def load_volume(name):
	img = nib.load(name).get_fdata()
	return img

def crop_volume_disp(volume,slicing_up=0.4,slicing_bot=0.5):

    roi = []
    left_cord = right_cord = bottom_cord = top_cord = []
    first_layer_checked = int(volume.shape[2] * 0.1)
    last_layer_checked = volume.shape[2] - int(volume.shape[2] * 0.1)

    norm_vol = cv2.normalize(volume, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    for layer in range(first_layer_checked, last_layer_checked):

        if np.max(norm_vol[:,:,layer]) > 55:
            left, bottom, right, top = compute_roi(get_brain_contour_nii(norm_vol[:,:,layer]))

            area = (right-left) * (top-bottom)
            left_cord.append(left)
            right_cord.append(right)
            bottom_cord.append(bottom)
            top_cord.append(top)
            roi.append(area)

    min_left = np.min(left_cord)
    max_right = np.max(right_cord)
    min_bottom = np.min(bottom_cord)
    max_top = np.max(top_cord)

    #compute the crop
    volume = volume[min_bottom : max_top, min_left : max_right, :]
    # print('volume shape is',volume.shape)
    return volume

def pred(volume_file):
    # headers={'Content-Type':'application/json'}
    # resp= requests.post(API_URL, json=volume_file,headers=headers).json()

    with requests.Session() as s:
        response = s.post(API_URL,files=volume_file)

    if response.status_code == 200:
        resp = response.json()

    else:
        resp = response.json()
        resp
        ":grimacing: api error :robot_face:"

    return resp

def image_to_dict(image_array, dtype='uint8', encoding='utf-8'):
    '''
    Convert an ndarray representing a batch of images into a compressed string
    ----------
    Parameters
    ----------
    imgArray: a np array representing an image
    ----------
    Returns
    ----------
    dict(image: str,
         height: int,
         width: int,
         channel: int)
    '''
    # Get current shape, only for single image
    if image_array.ndim < 2 or image_array.ndim > 4:
        raise TypeError
    elif image_array.ndim < 3:
        image_array.reshape(*image_array.shape, 1)
    elif image_array.ndim < 4:
        size = 1
        height, width, channel = image_array.shape
    elif image_array.ndim > 4:
        size, height, width, channel = image_array.shape

    # Ensure uint8
    image_array = image_array.astype(dtype)
    # Flatten image
    image_array = image_array.reshape(size * height * width * channel)
    # Encode in b64 for compression
    image_array = base64.b64encode(image_array)
    # Prepare image for POST request, ' cannot be serialized in json
    image_array = image_array.decode(encoding).replace("'", '"')

    api_dict = {'image': image_array, 'size': size, 'height': height,
                'width': width, 'channel': channel}

    return api_dict

#  ------ FRONT-END STARTS HERE ------

#with st.sidebar:
    #title='<p style="font-family:Courier; color:Blue; font-size: 18px;"> MRI info and params </p>'
    #st.markdown(title,unsafe_allow_html=True)

st.set_page_config(layout="wide")
original_title = '<p style="font-family:Courier; color:Blue; font-size: 35px;">Brain Signal</p>'
st.markdown(original_title, unsafe_allow_html=True)
image = Image.open('BrainSignal_.png')
st.image(image)


#menu = ["Image","Volume"]
#choice = st.sidebar.selectbox("Menu",menu)

st.markdown("<h3 style='text-align: center; color: #3a72cf;'>Detect Brain Pathologies in 3D MRI scans</h3>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: #1f54ac;'>Deep Learning - 3D Convolutional Neuronal Networks</h6>", unsafe_allow_html=True)

upload_mess="<h6 style='text-align: center; color: #1f54ac;'> Drop your brain 🧐</h6>"
st.markdown(upload_mess,unsafe_allow_html=True)
volume_file = st.file_uploader(' ')

if volume_file is not None:
    with open(volume_file.name, 'wb') as f:
        f.write(volume_file.getbuffer())

    volume=load_volume(volume_file.name)
    #volume_str=image_to_dict(volume)

    # convert image to bytes
    img_byte_arr2 = iioo.BytesIO()
    volume.load(img_byte_arr2, mm)
    img_byte_arr2 = img_byte_arr2.getvalue()

    with open("array.npy", "wb") as f:
        f.write(img_byte_arr2)


    # api call
    files = {"image": img_byte_arr2}


    with st.spinner('Preprocessing in progress ...🧘🏻‍♂️'):
        volume = crop_volume_disp(volume)
        volume = resize_and_pad(volume)
        volume_proc= normalize_vol(volume)
    processed_mess='<p style="font-family:Courier; color:Blue; font-size: 18px;">Processed & Upload achieved ✅ </p>'
    st.markdown(processed_mess,unsafe_allow_html=True)

    col11, col21, col31 = st.columns(3)
    bt = col21.button('Prediction')

    if bt:
        st.markdown('------------')
        st.markdown(pred(volume_str))
        st.markdown('------------')

    # col1, col2, col3 = st.columns([1,1,1])

    # with col1:
    #     st.button('Show processed 2D image')
    # with col2:
    #     st.button('Show 3D image')
    # with col3:
    #     st.button('Make Prediction')

    # if st.button('Show processed 2D image'):
    # # print is visible in the server output, not in the page
    #     c1,c2=st.columns((1,2))
    #     volume_initial =volume.shape
    #     shape_mess=f"""<style> p.a {{font-family: Arial;
    #             color:Black;
    #             font-size: 18px;}}
    #             </style>
    #             <ul>
    #             <li>Volume dimensions before preprocessing: {volume_initial}</li>
    #             </ul>
    #             """
    #     c1.markdown(shape_mess,unsafe_allow_html=True)

    #     volume_proc_val =volume_proc.shape
    #     shape_mess_proc=f"""<style> p.a {{font-family: Arial;
    #             color:Black;
    #             font-size: 18px;}}
    #             </style>
    #             <ul>
    #             <li>Volume dimensions before preprocessing: {volume_proc_val}</li>
    #             </ul>
    #             """
    #     c1.markdown(shape_mess_proc,unsafe_allow_html=True)

    #     Image_disp='<div style="text-align: center"><p style="font-family:Courier; color:Blue; font-size: 18px;">Processed Images </p></div>'
    #     c2.markdown(Image_disp,unsafe_allow_html=True)
    #     c2.display_image(volume)

    # if st.button('Show 3D image'):
    #     Image_disp='<div style="text-align: center"><p style="font-family:Courier; color:Blue; font-size: 18px;">3D plot</p></div>'
    #     st.markdown(Image_disp,unsafe_allow_html=True)
    #     #display volume
    #     st.display_volume(volume)

    # if st.button('Make Prediction'):
    #     Image_disp='<p style="font-family:Courier; color:Blue; font-size: 18px;">Pathological Condition</p>'
    #     st.markdown(Image_disp,unsafe_allow_html=True)
    #     st.markdown()




#if choice == "Image":
    #st.subheader("Image")
    #image_file = st.file_uploader("Upload", type=["png","jpg","jpeg"])

    #if image_file is not None:
        #st.image(load_image(image_file),width=250)
        #st.markdown(f''' {np.array(image_file)}''')
        #st.markdown(requests.post(API_URL, files={'file': image_file.getvalue()}).json())


#if choice == "Volume":
    #st.subheader("Volume")
    #volume_file = st.file_uploader("Upload")
    #volume_data = nib.load(volume_file).get_fdata()

    #if volume_file is not None:
        #with open(volume_file.name, 'wb') as f:
            #f.write(volume_file.getbuffer())
        #vol=load_volume_test(volume_file.name)

        # files = open(volume_file, 'rb')
        #st.markdown("--------------------")
        #st.markdown(requests.post(API_URL, files={'file': volume_file.getvalue()}).json())
        #st.markdown("--------------------")

        # with requests.Session() as s:
        #     r = s.post(API_URL,files=volume_file)
        #     st.markdown("--------------------")
        #     st.markdown(r.status_code)
        #     st.markdown("--------------------")

        # st.plotly_chart(plot3D(vol), use_container_width=False, sharing="streamlit")
