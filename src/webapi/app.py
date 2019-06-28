#!flask/bin/python
from flask import Flask
from flask import request
from flask import jsonify
import os
import io
import sys
import numpy as np
import base64
from PIL import Image
from keras.backend import clear_session
import tensorflow as tf

ROOT_DIR = os.path.abspath("./src/Mask_RCNN")
sys.path.append(ROOT_DIR)

from mrcnn import model
from mrcnn.config import Config

np.set_printoptions(threshold=sys.maxsize)

global _model
global _graph
app = Flask(__name__)

MODEL_NAME = "mask_rcnn_hq"

class InferenceConfig(Config):
    NAME="HQ"
    NUM_CLASSES=1+2
    GPU_COUNT=1
    IMAGES_PER_GPU=1

def load_model():
    global _model
    model_folder_path = os.path.abspath("./models/") + "/"
    _model = model.MaskRCNN('inference', InferenceConfig(), model_folder_path)
    _model.load_weights(os.path.abspath(model_folder_path + MODEL_NAME + ".hdf5"))
    global _graph
    _graph = tf.get_default_graph()

load_model()

@app.route('/v<int:version>/models/<string:model_name>:<string:action>', methods=['POST'])
def predict(version:int, model_name:str, action:str):
    if version != 1 or model_name != MODEL_NAME or action != "predict":
        return "Not implemented", 501

    b64_image = request.data
    image_data = base64.b64decode(b64_image)
    image_bytes = Image.open(io.BytesIO(image_data))
    image_array = np.array(image_bytes)
    global _model
    global _graph
    with _graph.as_default():
        results = _model.detect([image_array], verbose=1)
    return str(results)

if __name__ == '__main__':
    app.run(debug=True)