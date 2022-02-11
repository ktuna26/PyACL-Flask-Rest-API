"""
Redis server, it is used to run 
pyacl model asynchronously through redis 

Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-02-05 15:12:13
MODIFIED: 2022-02-10 01:48:45
"""
# -*- coding:utf-8 -*-
import numpy as np
import warnings

from os import path
from time import sleep
from model.acl import NET
from sys import version_info
from json import dumps, loads
from redis import StrictRedis
from base64 import decodestring
from configparser import ConfigParser


class RunNet(object):
    def __init__(self):
        self.model = None
        self.cfg = None
        self.redis_db = None
        
        self.__init_resource__()

    
    def __init_resource__(self):
        # ignore python warnings
        warnings.filterwarnings("ignore")
        
        # define configurations
        print("[INFO] loading configurations . . .")
        self.cfg = ConfigParser()
        self.cfg.read(path.abspath('./data/app.cfg'))

        # connect to Redis server
        self.redis_db = StrictRedis(host=self.cfg.get('db-server', 'host'),
                                    port=self.cfg.getint('db-server', 'port'), 
                                    db=self.cfg.getint('db-server', 'db_num'))
        
        # initialize models
        device_id = self.cfg.getint('npu', 'device_id')
        print("[INFO] using NPU-%d . . ."% device_id)
        model_path = path.abspath(self.cfg.get('model', 'path'))
        self.model = NET(device_id, model_path) # load model
        print("[INFO] type of the model ", str(type(self.model)))
        
        # serialize model input info from getting pyacl model
        w, h = self.model.get_model_input_dims()
        d_model_input_dims = {'w': w, 'h': h}
        self.redis_db.rpush('model_input_dims', dumps(d_model_input_dims))


    def __base64_decode_image(self, a, dtype, shape):
        # if this is Python 3, we need the extra step of encoding the
        # serialized NumPy string as a byte object
        if version_info.major == 3:
            a = bytes(a, encoding="utf-8")
            
        # convert the string to a NumPy array using the supplied data
        # type and target shape
        a = np.frombuffer(decodestring(a), dtype=dtype)
        a = a.reshape(tuple(shape))
        
        # return the decoded image
        return a

    
    def model_process(self):
        # continually pool for new images to process
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            queue = self.redis_db.lrange('model_img_queue', 0, 
                    self.cfg.getint('model', 'batch_size') - 1)
            
            images = []
            imageIDs = []
            imageDims = []
            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                q = loads(q.decode("utf-8"))
                image = self.__base64_decode_image(q["img"], self.cfg.get('model', 'dtype'),
                                                    q["img_np_dims"])
                
                # update list of image dims
                imageDims.append(q["img_dims"])
                # update list of image
                images.append(image)
                # update the list of image IDs
                imageIDs.append(q["id"])
            
            # output = []
            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                # classify the batch
                results = self.model.run(images, imageDims)
                
                # loop over the image IDs and their corresponding set of
                # results from our model
                for (imageID, result) in zip(imageIDs, results):
                    # initialize the list of output predictions
                    r = {"bboxes": str(result)}
                    
                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    self.redis_db.set(imageID, dumps(r))
                    
                # remove the set of images from our queue
                self.redis_db.ltrim('model_img_queue', len(imageIDs), -1)
                
            # sleep for a small amount
            sleep(self.cfg.getfloat('test', 'sleep'))


# pyacl_app main
if __name__ == "__main__":
    runNet = RunNet()
    runNet.model_process()