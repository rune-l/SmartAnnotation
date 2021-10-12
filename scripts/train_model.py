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
from src.maskrcnn.config import Config
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

class ADE20KConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ADE20K"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3405 # background + 3406 shapes from the data.

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


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
            epochs=1, 
            layers='heads')
