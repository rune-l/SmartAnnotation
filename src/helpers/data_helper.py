"""
Helper functions needed to handle the images.
Helper functions are for both the image and the metadata for masks.
"""

# Import packages #
import json
import cv2
import os
import numpy as np
import pandas as pd
from gluoncv.data import ADE20KSegmentation
from gluoncv.utils.viz import get_color_pallete
from PIL import Image, ImageDraw
from typing import List, Tuple


# Write helper functions #
def find_metadata_files(directory: str, extension: str) -> List[str]:
    """Find all files in data directory with certain extension."""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] == extension:
                fullpath = os.path.join(root, file)
                file_list.append(fullpath)
    return file_list

def get_metadata(meta_filename: str) -> dict:
    """
    Return the needed information from the metadata file.
    Input:Takes the metadata file as input
    Output: Returns a dict with the following information.
    - img_file: str = Image file name.
    - img_size: list = List of image size.
    - objects: The objects in the in the image, contains both the class and the polygon coordinates.
    - parts: The same as ojects but for the parts (smaller objects, such as a building window).
    """
    # Define empty dicts to store results on #
    objects = {}
    parts = {}

    with open(meta_filename, 'r') as f:
        input_info = json.load(f)

    contents = input_info['annotation']['object']
    instance = np.array([int(x['id']) for x in contents])
    names = [x['raw_name'] for x in contents]
    corrected_raw_name =  [x['name'] for x in contents]
    partlevel = np.array([int(x['parts']['part_level']) for x in contents])
    ispart = np.array([p>0 for p in partlevel])
    iscrop = np.array([int(x['crop']) for x in contents])
    listattributes = [x['attributes'] for x in contents]
    polygon = [x['polygon'] for x in contents]
    for p in polygon:
        p['x'] = np.array(p['x'])
        p['y'] = np.array(p['y'])

    objects['instancendx'] = instance[ispart == 0]
    objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
    objects['corrected_raw_name'] = [
        corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])
    ]
    objects['iscrop'] = iscrop[ispart == 0]
    objects['listattributes'] = [
        listattributes[x] for x in list(np.where(ispart == 0)[0])
    ]
    objects['polygon'] = [
        polygon[x] for x in list(np.where(ispart == 0)[0])
    ]

    parts['instancendx'] = instance[ispart == 1]
    parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
    parts['corrected_raw_name'] = [
        corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])
    ]
    parts['iscrop'] = iscrop[ispart == 1]
    parts['listattributes'] = [
        listattributes[x] for x in list(np.where(ispart == 1)[0])
    ]
    parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    # Define the output dict #
    meta_dict = {
        'img_file': input_info['annotation']['filename'],
        'img_size': input_info['annotation']['imsize'],
        'objects': objects,
        'parts': parts
    }

    return meta_dict

def create_mask_array(x_arr: np.ndarray, y_arr: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Create the image mask for the polygon. Here the image mask is an array of zeros and ones"""
    
    # Create list of coordinates for the polygon edges #
    poly_verts = [(xc, yc) for xc, yc in zip(x_arr, y_arr)]

    # Create new image of zeros #
    img = Image.new('L', (img_width, img_height), 0)

    # Draw the polygon with ones #
    ImageDraw.Draw(img).polygon(poly_verts, outline=1, fill=1)
    mask = np.array(img)
    
    return mask


def resize_mask_array(mask_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize img to conform to standard image size"""
    resized_img = cv2.resize(mask_array, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

def prepare_mask_data(meta_data: dict, target_size: Tuple[int, int], config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the labels and their placements for the image.
    Consists of an array of class_ids and the array of masks.
    The mask array will have the size (target_size[0], target_size[1], n_objects_in_image).
    Each mask will be a boolean array.
    """

    id_list = []
    mask_list = []

    img_size = meta_data['img_size']

    for name, polygon in zip(meta_data['objects']['corrected_raw_name'], meta_data['objects']['polygon']):
        
        # Extract object name and id #
        obj_class = name
        obj_id = config[obj_class]['idx']
        
        # Extract polygon coordinates #
        poly_x = polygon['x']
        poly_y = polygon['y']

        # Create and resize layer mask #
        poly_mask = create_mask_array(x_arr=poly_x, y_arr=poly_y, img_width=img_size[0], img_height=img_size[1])
        resized_mask = resize_mask_array(mask_array=poly_mask, target_size=target_size)

        # Append results #
        id_list.append(obj_id)
        mask_list.append(resized_mask.astype(bool))

    # Create output #
    id_array = np.array(id_list)
    mask_array = np.dstack(mask_list)

    return id_array, mask_array

if __name__ == '__main__':
    example_file = './data/example.json'
    config_file = './data/code_config.json'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    n_obj_in_example = 51

    metadata = get_metadata(meta_filename=example_file)

    target_size = (128, 128)

    id_array, mask_array = prepare_mask_data(meta_data=metadata, target_size=target_size, config=config)

    print('First five ids: {}'.format(id_array[0:5]))
    print('Size of mask array: {}'.format(mask_array.shape))

