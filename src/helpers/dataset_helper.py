"""
Create the ADE20K dataset so that it can go into the model.
"""

# Import packages #
import os
import json
import numpy as np
import pickle as pkl
from typing import Tuple
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Define custom functions #
import src.helpers.data_helper as data_helper
import src.helpers.utils as utils

# Define the dataset #
class ADE20K(utils.Dataset):
    """Generates ADE20K dataset."""
    def __init__(self, root_dir: str, config: dict, target_size: Tuple[int, int], index_file: str, train_flag=True):
        super().__init__()
        self.root_dir = root_dir
        self.config = config
        self.target_size = target_size
        self.source = "ADE20K"
        self.train_flag = train_flag
        
        with open(index_file, 'rb') as f:
            self.index_ade20k = pkl.load(f)


        if self.train_flag:
            check_str = 'training'
        else:
            check_str = 'validation'

        self.image_numbers = [
            i for i in range(0, len(self.index_ade20k['filename'])) if 
            check_str in self.index_ade20k['folder'][i]
        ]

    
    def load_image(self, image_id: int) -> np.ndarray:
        """Load the image with the given target size."""
        file_name: str = self.index_ade20k['filename'][image_id]
        file_folder: str = self.index_ade20k['folder'][image_id]
        folder_path = os.path.join(self.root_dir, file_folder)
        file_path = os.path.join(folder_path, file_name)

        # Load the image #
        img = load_img(path=file_path, target_size=self.target_size)

        # Convert to numpy array #
        img_arr = img_to_array(img)

        return img_arr

    def load_mask(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load the mask. Returns an array of class ids and a numpy array of the boolean array masks."""
        file_name: str = self.index_ade20k['filename'][image_id]
        file_folder: str = self.index_ade20k['folder'][image_id]
        meta_file = file_name.replace('.jpg', '.json')
        folder_path = os.path.join(self.root_dir, file_folder)
        
        meta_path = os.path.join(folder_path, meta_file)

        metadata = data_helper.get_metadata(meta_filename=meta_path)

        class_ids, masks = data_helper.prepare_mask_data(meta_data=metadata, target_size=self.target_size,
                                                         config=self.config, img_folder=folder_path)

        return masks, class_ids.astype(np.int32)

    def load_all_images(self):
        
        # Add the classes #
        for key in self.config.keys():
            self.add_class(
                source=self.source,
                class_id=self.config[key],
                class_name=key
            )

        # Add images by their image id (place in file list) #
        for image_id in self.image_numbers:
            
            file_name: str = self.index_ade20k['filename'][image_id]
            file_folder: str = self.index_ade20k['folder'][image_id]
            folder_path = os.path.join(self.root_dir, file_folder)
            file_path = os.path.join(folder_path, file_name)

            self.add_image(
                source=self.source,
                image_id=image_id,
                path=file_path
            )


if __name__ == '__main__':
    
    data_dir = os.path.join(os.getcwd(), 'data', 'ADE20K_2021_17_01')
    config_path = os.path.join(os.getcwd(), 'data', 'code_config.json')
    target_size = (256, 256)
    index_file = os.path.join(os.getcwd(), 'data', 'ADE20K_2021_17_01', 'index_ade20k.pkl')
    

    with open(config_path) as fc:
        config_dict = json.load(fc)

    train_data = ADE20K(
        root_dir=data_dir,
        config=config_dict,
        target_size=target_size,
        index_file=index_file,
        train_flag=True
    )

    val_data = ADE20K(
        root_dir=data_dir,
        config=config_dict,
        target_size=target_size,
        index_file=index_file,
        train_flag=False
    )

    train_data.load_all_images()
    train_data.prepare()

    val_data.load_all_images()
    val_data.prepare()

    # Print #
    print('Training data initialized with {} images.'.format(len(train_data.image_ids)))
    print('Validation data initialized with {} images.'.format(len(val_data.image_ids)))
    print('Number of classes: {}'.format(train_data.num_classes))

