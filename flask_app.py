"""
PyACL Flask Rest-API
Copyright 2022 Huawei Technologies Co., Ltd

Usage:
  $ python3 app.py

CREATED:  2021-11-24 15:12:13
MODIFIED: 2022-02-10 00:48:45
"""
# -*- coding:utf-8 -*-
import numpy as np
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

from os import path
from PIL import Image
from time import sleep
from uuid import uuid4
from io import BytesIO
from model.acl import NET
from base64 import b64encode
from redis import StrictRedis
from configparser import ConfigParser
from utils.preprocessing import preprocess
from flask import Flask, json, Response, request
from flask_restplus import Api, Resource, reqparse


# initialize flask app
app = Flask(__name__)

# define configurations
print("[INFO] loading configurations . . .")
app.cfg = ConfigParser()
app.cfg.read(path.abspath('./data/app.cfg'))


# return the succes message with api
def error_handle(output, status=500, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

# return the error message with api
def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


# base64 encoding
def base64_encode_image(a):
    # base64 encode the input NumPy array
    return b64encode(a).decode("utf-8")

# run yolov5 detector
def run_detector(img, model_input_size):
    print("[INFO] running model . . .")

    # convert RGB PIL image to RGB Cv2 image
    img_rgb = np.array(img.convert('RGB'))
    # convert img to data
    img_data = preprocess(img_rgb, model_input_size)
            
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    img_data = img_data.copy(order="C")
            
    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid4())
    img_data_dims = img_data.shape
    img_data = base64_encode_image(img_data)
    d = {"id": k, "img": img_data, "img_dims" : img_rgb.shape, 
        "img_np_dims" : img_data_dims}
    app.redis_db.rpush('model_img_queue', json.dumps(d))
            
    # keep looping until our model server returns the output
    # predictions
    bboxes = ""
    while True:
        # attempt to grab the output predictions
        output = app.redis_db.get(k)
                
        # check to see if our model has classified the input
        # image
        if output is not None:
            # add the output predictions to our data
            # dictionary so we can return it to the client
            output = output.decode("utf-8")
            bboxes = json.loads(output)
                    
            # delete the result from the database and break
            # from the polling loop
            app.redis_db.delete(k)
            break
                    
        # sleep for a small amount to give the model a chance
        # to classify the input image
        sleep(app.cfg.getfloat('test', 'sleep'))

    print("[RESULT] bboxes --> ", bboxes)
    return bboxes


# initialize flask restful app
resutfulApp = Api(app = app, 
                  version = app.cfg.get('swagger', 'version'), 
                  title = app.cfg.get('swagger', 'title'), 
                  description = app.cfg.get('swagger', 'description1'))
name_space = resutfulApp.namespace('analyze', description = app.cfg.get('swagger', 'description2'))


#ModelService swagger settings
model_service_param_parser = reqparse.RequestParser()
model_service_param_parser.add_argument('image',  
                         type=werkzeug.datastructures.FileStorage, 
                         location='files', 
                         required=True, 
                         help='Image file')

# run model
@name_space.route('', methods = ['POST'])
@name_space.expect(model_service_param_parser)
@resutfulApp.doc(responses={
        200: 'Success',
        400: 'Validation Error'
    },description = app.cfg.get('swagger', 'description3')
    )
class ModelService(Resource):
    def post(self):
        # check image is uploaded with image keyword
        if 'image' not in request.files:
            print("[ERROR] image required")
            return error_handle(json.dumps({"errorMessage" : "image required"}), 400)
        
        image = request.files['image']
        print("[INFO] image : %s"% image)
        
        # chech extension of image
        filename = image.filename
        if '.' in filename and \
        filename.rsplit('.', 1)[1].lower() not in app.allowed_extensions:
            return error_handle(json.dumps({"errorMessage" : "image format must be one of " + \
                                app.allowed_extensions}), status=400)
        
        # read the image in PIL format
        print("[INFO] loading image . . .")
        img = image.read()
        # convert image format
        try:
            img = Image.open(BytesIO(img))
        except:
            return error_handle(json.dumps({"errorMessage" : "Uploaded file is not a valid image"}), 
                                        status=400)
        
        # check min resolution
        width, height = img.size
        if width < app.min_width or height < app.min_height :
            return error_handle(json.dumps({"errorMessage" : "Image resolution must be greater than 300x300"}), 
                                        status=400)

        # deserialize the text-detection model output info
        q_model_input_dims = json.loads(app.redis_db.lrange('model_input_dims', 0, 2)[0].decode("utf-8"))
        boxes_coord = run_detector(img, (q_model_input_dims['w'], q_model_input_dims['h']))

        return success_handle(json.dumps(boxes_coord))


def init():
    # connect to Redis server
    print("[INFO] initialize the redis server . . .")
    app.redis_db = StrictRedis(host=app.cfg.get('db-server', 'host'),
                            port=app.cfg.getint('db-server', 'port'), 
                            db=app.cfg.getint('db-server', 'db_num'))
    
    # define allowed file types
    app.allowed_extensions = app.cfg.get('file', 'allowed_extensions')

    # define allowed min image size
    app.min_width = app.cfg.getint('file', 'min_width')
    app.min_height = app.cfg.getint('file', 'min_height')

# run api 
if __name__ == "__main__":
    print("[INFO] strating ocr_api . . .")
    init()
        
    app.run(host = app.cfg.get('rest-server', 'host'), 
            port = app.cfg.get('rest-server', 'port'),
            threaded = app.cfg.get('rest-server', 'threaded'),
            debug = app.cfg.getboolean('rest-server', 'debug'))