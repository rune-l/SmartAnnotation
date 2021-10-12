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
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import List, Tuple

# Constant #
ENCODING = 'iso-8859-1'

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
    - full_path: str = Full image path.
    - img_size: list = List of image size.
    - objects: The objects in the in the image, contains both the class and the polygon coordinates.
    - parts: The same as ojects but for the parts (smaller objects, such as a building window).
    """
    # Define empty dicts to store results on #
    objects = {}
    parts = {}

    with open(meta_filename, 'r', encoding=ENCODING) as f:
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
    mask_path = [x['instance_mask'] for x in contents]
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
    objects['mask_path'] = [
        mask_path[x] for x in list(np.where(ispart == 0)[0])
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
    parts['mask_path'] = [mask_path[x] for x in list(np.where(ispart == 1)[0])]

    # Define the output dict #
    meta_dict = {
        'img_file': input_info['annotation']['filename'],
        'full_path': meta_filename.replace('.json', '.jpg'),
        'img_size': input_info['annotation']['imsize'],
        'objects': objects,
        'parts': parts
    }

    return meta_dict

def create_mask_array(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Create the image mask for the polygon. Here the image mask is an array of zeros and ones"""
    
    # Load image #
    img = load_img(path=mask_path, target_size=target_size)
    
    # Convert to a numpy array #
    arr = img_to_array(img)

    # Sum over the RGB axis #
    arr = arr.sum(axis=-1)

    # Initiate the mask array #
    mask_arr = np.zeros(arr.shape)

    # Set all the white pixels to 1 #
    mask_arr[arr >= (255*3)] = 1

    # Convert mask to boolean array #
    mask_arr = mask_arr.astype(bool)
    
    return mask_arr

def resize_mask_array(mask_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize img to conform to standard image size"""
    resized_img = cv2.resize(mask_array, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    return resized_img

def prepare_mask_data(meta_data: dict, target_size: Tuple[int, int], config: dict, img_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the labels and their placements for the image.
    Consists of an array of class_ids and the array of masks.
    The mask array will have the size (target_size[0], target_size[1], n_objects_in_image).
    Each mask will be a boolean array.
    """

    id_list = []
    mask_list = []

    for name, mask_path in zip(meta_data['objects']['corrected_raw_name'], meta_data['objects']['mask_path']):
        
        # Extract object name and id #
        obj_class = name
        obj_id = config[obj_class]['idx']
        
        # Define full mask path #
        mask_full_path = os.path.join(img_folder, mask_path)

        # Create and resize layer mask #
        poly_mask = create_mask_array(mask_path=mask_full_path, target_size=target_size)

        # Append results #
        id_list.append(obj_id)
        mask_list.append(poly_mask)

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
