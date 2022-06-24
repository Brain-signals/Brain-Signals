import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import plotly.graph_objects as go

def NII_image_shape(path):
    test_load = nib.load(f'{path}').get_fdata()
    return test_load.shape

def show_nii_2D(volume):

    slice_x = int(volume.shape[0]/2)
    slice_y = int(volume.shape[1]/2)
    slice_z = int(volume.shape[2]/2)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(10,4))
    ax1.imshow(volume[slice_x,:,:])
    ax1.set_title('X axis')
    ax1.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax2.imshow(volume[:,slice_y,:])
    ax2.set_title('Y axis')
    ax2.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    ax3.imshow(volume[:,:,slice_z])
    ax3.set_title('Z axis')
    ax3.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)

    print(f'shape is {volume.shape}')

    return fig

def show_nii_3D(volume):
  volume = ndimage.zoom(volume,np.array([96/volume.shape[0], 96/volume.shape[1], 1]))

  # Define frames
  nb_frames = volume.shape[-1] - 1 #minus 1 frame for initial display

  fig = go.Figure(frames=[go.Frame(data=go.Surface(z=(nb_frames/10 - k * 0.1) * np.ones((96, 96)),
                                                   surfacecolor=np.rot90(volume[:,:,nb_frames - k]),
                                                   cmin=0, cmax=255
                                                   ),
                                   name=str(k) # you need to name the frame for the animation to behave properly
                                   )
                  for k in range(20,nb_frames-10)])

  # Add data to be displayed before animation starts
  fig.add_trace(go.Surface(
      z=nb_frames/10 * np.ones((96, 96)),
      surfacecolor=np.rot90(volume[:,:,nb_frames]),
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
