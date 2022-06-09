
import nibabel as nib
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2

import streamlit as st
from scipy import ndimage

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
                  for k in range(16,nb_frames)])

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

def bright_contrast(img):
      #create trackbar for brightness
    cv2.createTrackbar('Brightness','GEEK', 255, 2*255, BrightnessContrast)
    #contrast range -127 to 127
    cv2.createTrackbar('Contrast', 'GEEK', 127, 2*127, BrightnessContrast)
    BrightnessContrast(img,0)

    cv2.waitKey(0)

def BrightnessContrast(img,brightness=0):

    # getTrackbarPos returns the
    # current position of the specified trackbar.
    brightness = cv2.getTrackbarPos('Brightness',
                                    'GEEK')

    contrast = cv2.getTrackbarPos('Contrast',
                                  'GEEK')

    effect = controller(img,
                        brightness,
                        contrast)

    # The function imshow displays
    # an image in the specified window
    cv2.imshow('Effect', effect)

def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted
        # calculates the weighted sum
        # of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified
    # text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    return cal

def load_volume(name):
	img = nib.load(name).get_fdata()
	return img

def gammaCorrection(img_original,gamma):
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(img_original, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv2.hconcat([img_original, res])
    cv2.imshow("Gamma correction", img_gamma_corrected)
    return img_gamma_corrected
