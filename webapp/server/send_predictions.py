# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:37:54 2021

@author: Yassine
"""

# -*- coding: utf-8 -*-

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np

from detectron2 import model_zoo

import sys
import os


def score_image(predictor: DefaultPredictor, image_url: str):
    # load an image of Lionel Messi with a ball
    image_reponse = requests.get(image_url)
    image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

    # make prediction
    return predictor(image)

def prepare_predictor():
    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    cfgFile = 'c:\\users\\yassine\\downloads\\maskformer-main\\detectron2-main\\detectron2\\model_zoo\\configs\\COCO-Detection\\faster_rcnn_R_101_FPN_3x.yaml'
    cfg.merge_from_file(cfgFile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu" # we use a CPU Detectron copy
    

    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    
    return (predictor, classes)


def predict_image(img, predictor, classes):
    outputs = predictor(img)
    bboxes = outputs["instances"].pred_boxes
    bboxes = bboxes.tensor.numpy()
    bboxes = [list(bb) for bb in bboxes]
    
    labels = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist() 
    
    js_pred =  {'bbox': [], 
                'class': [], 
                'score': []}

    for i in range(len(labels)):
        js_pred['bbox'].append(bboxes[i])
        js_pred['class'].append(classes[labels[i]])
        js_pred['score'].append(scores[i])
        
        
    return js_pred


if __name__ == "__main__": 
    print(os.getcwd())

    img  = cv2.imread("./img.jpg")
    predictor, classes = prepare_predictor()
    js_pred = predict_image (img, predictor, classes)
    
    print(js_pred)
    

