# Import packages #
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
from src.helpers import utils
import src.maskrcnn.model as modellib
from src.maskrcnn import visualize
from src.maskrcnn.model import log

# Import dataset helper #
import src.helpers.dataset_helper as dataset_helper

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'models', "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


if __name__ == '__main__':
    config = ADE20KConfig()

    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

    # Load weights #
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

    # Initialize datasets #
    data_dir = os.path.join(os.getcwd(), 'data')
    config_path = os.path.join(os.getcwd(), 'data', 'code_config.json')
    target_size = (256, 256)
    index_file = os.path.join(os.getcwd(), 'data', 'ADE20K_2021_17_01', 'index_ade20k.pkl')
    

    with open(config_path) as fc:
        config_dict = json.load(fc)

    train_data = dataset_helper.ADE20K(
        root_dir=data_dir,
        config=config_dict,
        target_size=target_size,
        index_file=index_file,
        train_flag=True
    )

    val_data = dataset_helper.ADE20K(
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

    print('Start model training...')
    model.train(train_data, val_data, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')
