import numpy as np
import cv2
import os
from scipy import ndimage



def compute_roi(contour):
    l = [dots.tolist()[0] for dots in contour]
    xs, ys = zip(*l)
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)



def get_brain_contour_nii(img):

    # convert nifti slice to cv2 image
    img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # compute maximum area
    max_area = img.shape[0] * img.shape[1]

    tresh_params, thresh = cv2.threshold(img, 10, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    areas = []
    for c in contours:
        # retrieve bounding box
        left, bottom, right, top = compute_roi(c)
        # print(left,right,bottom,top)

        # compute ROI area => the biggest wins
        area = (right-left) * (top-bottom)
        if area < max_area:
            areas.append(area)

    return contours[np.argmax(areas, axis=0)]



def crop_volume(volume,slicing_up=0.4,slicing_bot=0.3):

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

    index_max_area = roi.index(np.max(roi))
    top_z = index_max_area + int(volume.shape[2]*slicing_up)
    bottom_z = index_max_area - int(volume.shape[2]*slicing_bot)
    # print('top_z is',top_z)
    # print('bottom_z is',bottom_z)

    if top_z > volume.shape[2]:
      top_z = volume.shape[2] - int(volume.shape[2]*0.2)
    if bottom_z < 0:
      bottom_z = 0 + int(volume.shape[2]*0.2)

    return volume[:, :, bottom_z:top_z]




def resize_and_pad(volume):

    target_res = int(os.environ.get("TARGET_RES"))

    # Get current shape
    current_width = volume.shape[0]
    current_length = volume.shape[1]
    current_depth = volume.shape[2]

    # Compute shape factor
    width_factor = 1 / ( current_width / target_res )
    length_factor = 1 / ( current_length / target_res )
    depth_factor = 1 / ( current_depth / target_res )

    factor = min(width_factor,length_factor,depth_factor)

    # Zoom to the target, based on the biggest axis
    volume = ndimage.zoom(volume, (factor, factor, factor))

    # and pad zeros until wanted shape

    def get_padding(axis_shape):
        zeros_to_add = target_res-axis_shape
        if zeros_to_add%2 == 0:
            padding = (int(zeros_to_add/2),int(zeros_to_add/2))
        else:
            padding = (int(zeros_to_add//2),int(zeros_to_add//2+1))
        return padding

    pad_width = get_padding(volume.shape[0])
    pad_length = get_padding(volume.shape[1])
    pad_depth = get_padding(volume.shape[2])

    volume = np.pad(volume, (pad_width, pad_length, pad_depth), mode='constant')

    return volume



def normalize_vol(X):
    """Normalize the volume"""
    return cv2.normalize(X, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
