# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 00:00:17 2021

@author: Yassine
"""
import numpy as np
from imantics import Mask
from detectron2.config import get_cfg
from mask_former import add_mask_former_config
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask_former import add_mask_former_config


args = {"config_file": './app/server/ade20k-150/maskformer_R50_bs16_160k.yaml',
        "opts": ['MODEL.WEIGHTS', './app/server/model_final_d8dbeb.pkl']}
import os
def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    print(os.getcwd())
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg

def get_segmentation(path="img.png"):
    cfg =  setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    img = read_image(path, format="BGR")
    
    predictions = predictor(img)
    segmentation = predictions["sem_seg"].argmax(dim=0).cpu().detach()
    
    return (segmentation.numpy(), img.shape
)

def reduce_coords(coords, dimensions, thresh=10):
    thresh = np.sqrt(dimensions[0]*dimensions[1] / 256**2) * 10
    new_coords = [coords[0]]
    for point in coords:
        dist = np.linalg.norm(point - new_coords[-1])
        if dist > thresh:
            new_coords.append(point)
    
    return np.array(new_coords)

def get_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_polygons(segmention, dimensions):
    area_thresh = dimensions[0]*dimensions[1] / 256**2 * 100#00
    labels_ = np.unique(segmention)
    vertices = []
    for label in labels_:
        b = (segmention == label)
        c = b.astype(int)
        polygons = Mask(c).polygons()
        for poly in polygons.points:
            coords = reduce_coords(poly, dimensions)
            area = get_area(coords[:, 0], coords[:, 1])
            # if len(coords)>4 and area >area_thresh:
            if area >area_thresh:
                vertices.append((coords, label))
    
    return vertices

