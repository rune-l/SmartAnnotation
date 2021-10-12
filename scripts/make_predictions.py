"""Make predictions using the last trained model."""
import os
import json
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
from src.maskrcnn.config import ADE20KConfig
import src.helpers.utils as utils
import src.maskrcnn.model as modellib
import src.maskrcnn.visualize as visualize
from src.maskrcnn.model import log


# Import dataset helper #
import src.helpers.dataset_helper as dataset_helper

class InferenceConfig(ADE20KConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


if __name__ == '__main__':
    config = InferenceConfig()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=MODEL_DIR)

    # Load weights #
    model_path = model.find_last()

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Initialize datasets #
    data_dir = os.path.join(os.getcwd(), 'data')
    config_path = os.path.join(os.getcwd(), 'data', 'code_config.json')
    target_size = (256, 256)
    index_file = os.path.join(os.getcwd(), 'data', 'ADE20K_2021_17_01', 'index_ade20k.pkl')
    

    with open(config_path) as fc:
        config_dict = json.load(fc)

    val_data = dataset_helper.ADE20K(
        root_dir=data_dir,
        config=config_dict,
        target_size=target_size,
        index_file=index_file,
        train_flag=False
    )

    val_data.load_all_images()
    val_data.prepare()

    # Get random image to make prediction for #
    img_id = random.choice(val_data.image_ids)

    # Get image info #
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        val_data, config, 
        img_id
    )

    # Visualize original #
    print('Display true image objects:')
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
    # Make prediction #
    print('Make prediction for image: {}'.format(img_id))
    results = model.detect([original_image], verbose=1)
    r = results[0]
    
    fig_ax = get_ax(rows=1, cols=2, size=8)

    # Display differences #
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                val_data.class_names, figsize=(8, 8), ax=fig_ax[0], title='Original with mask')
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                val_data.class_names, r['scores'],
                                figsize=(8,8), ax=fig_ax[1], title='Model predictions')
    plt.show()
