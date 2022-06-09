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


API_URL = "http://127.0.0.1:8000/predict"


def plot3D(img):
  vol = img


  L=[]
  for i in range(vol.shape[-1]):
    if int(vol[:,:,i].sum())!=0:
      L.append(i)
  L = np.array(L)
  min_L = L.min()
  max_L = L.max()




  volume = vol.T
  r, c = volume[0].shape

  # Define frames
  import plotly.graph_objects as go
  nb_frames = vol.shape[-1]-1

  fig = go.Figure(frames=[go.Frame(data=go.Surface(
      z=(nb_frames/10 - k * 0.1) * np.ones((r, c)),
      surfacecolor=np.flipud(volume[nb_frames - k]),
      cmin=0, cmax=255
      ),
      name=str(k) # you need to name the frame for the animation to behave properly
      )
      for k in range(min_L,max_L,2)])

  # Add data to be displayed before animation starts
  fig.add_trace(go.Surface(
      z=nb_frames/10 * np.ones((r, c)),
      surfacecolor=np.flipud(volume[nb_frames]),
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

  fig.show()
  return fig




st.markdown('''
**This is a test**
''')


menu = ["Image","Volume"]
choice = st.sidebar.selectbox("Menu",menu)


def load_image(image_file):
	img = Image.open(image_file)
	return img

if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload", type=["png","jpg","jpeg"])

    if image_file is not None:
        st.image(load_image(image_file),width=250)
        st.markdown(f''' {np.array(image_file)}''')
        st.markdown(requests.post(API_URL, files={'file': image_file.getvalue()}).json())



def load_volume_test(name):
	img = nib.load(name).get_fdata()
	return img

if choice == "Volume":
    st.subheader("Volume")
    volume_file = st.file_uploader("Upload")
    #volume_data = nib.load(volume_file).get_fdata()

    if volume_file is not None:
        with open(volume_file.name, 'wb') as f:
            f.write(volume_file.getbuffer())
        vol=load_volume_test(volume_file.name)

        # files = open(volume_file, 'rb')
        st.markdown("--------------------")
        st.markdown(requests.post(API_URL, files={'file': volume_file.getvalue()}).json())
        st.markdown("--------------------")

        # with requests.Session() as s:
        #     r = s.post(API_URL,files=volume_file)
        #     st.markdown("--------------------")
        #     st.markdown(r.status_code)
        #     st.markdown("--------------------")

        # st.plotly_chart(plot3D(vol), use_container_width=False, sharing="streamlit")
