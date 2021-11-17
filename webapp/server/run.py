import cv2
from flask import Flask, Response  # Import flask
from random import *
from flask import jsonify
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from flask.wrappers import Request
import numpy as np

from detectron2 import model_zoo
import io
import json
import requests
from flask.globals import request
from PIL import Image

from get_segmentation import get_segmentation, get_polygons

def prepare_predictor():
    # create config
    cfg = get_cfg()
    # below path applies to current installation location of Detectron2
    # cfgFile = 'c:\\users\\yassine\\downloads\\maskformer-main\\detectron2-main\\detectron2\\model_zoo\\configs\\COCO-Detection\\faster_rcnn_R_101_FPN_3x.yaml'
    cfgFile = './app/server/faster_rcnn_R_101_FPN_3x.yaml'
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
 
    
    js_pred = []
    for i in range(len(labels)):
        predictions = {}
        boxes = bboxes[i]
        boxes[2] = boxes[2] - boxes[0]
        boxes[3] = boxes[3] - boxes[1]
        predictions['bbox'] = boxes
        predictions['class'] = classes[labels[i]]
        predictions['score'] = scores[i]
        js_pred.append(predictions)
        
    return js_pred

def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img

app = Flask(__name__, static_url_path='')  # Setup the flask app by creating an instance of Flask


@app.route('/upload', methods=['POST'])
def handle_form():
    f = request.files['image']
    f.save("./app/server/img.png")
    f.close
    print("saved")
    return jsonify({
        'success': True,
        'file': 'Received'
    })

@app.route('/prediction', methods=["GET", "POST"])
def get_prediction():
    predictor, classes = prepare_predictor()
    img = cv2.imread("./app/server/img.png")
    js_pred = predict_image(img, predictor, classes)
    
    return json.dumps(eval(str(js_pred))) #,  mimetype='application/json')


@app.route('/mask')
def get_mask():
    classes = np.load("./app/server/seg_classes.npy")
    segmentation, dimensions = get_segmentation("./app/server/img.png")
    vertices = get_polygons(segmentation, dimensions)
    js_pred = []
    for i in range(len(vertices)):
        predictions = {}
        
        x = vertices[i][0][:, 0] #/ dimensions[0] * 2199
        y = vertices[i][0][:, 1] #/ dimensions[1] * 1643
        nodes = [{"x": x[j],"y":y[j]} for j in range(len(x))]
        
        predictions['vertices'] = nodes
        predictions['class'] = classes[vertices[i][1]]
        predictions['score'] = 0.8
        js_pred.append(predictions)

    return json.dumps(eval(str(js_pred)))


@app.route('/')  # When someone goes to / on the server, execute the following function
def home():
    # return 'Hello, World!'  # Return this message back to the browser
    return app.send_static_file('index.html')

if __name__ == '__main__':  # If the script that was run is this script (we have not been imported)
    app.run(host='0.0.0.0', port=80)  # Start the server