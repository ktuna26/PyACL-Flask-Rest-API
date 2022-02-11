"""
This script is used to test flask rest api via multithreading

Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2022-02-10 01:12:13
MODIFIED: 2022-02-10 14:48:45
"""
# -*- coding:utf-8 -*-
# import the necessary packages
from os import path
from time import sleep
from requests import post
from threading import Thread
from configparser import ConfigParser

# define configurations
print("[INFO] loading configurations . . .")
cfg = ConfigParser()
cfg.read(path.abspath('./data/app.cfg'))


def call_predict_endpoint(n):
    # load the input image and construct the payload for the request
    image = open(cfg.get('test', 'img_path'), "rb")
    payload = {"image": image}
    # submit the request
    r = post(cfg.get('test', 'url'), files=payload).json()
    print(r)

    # ensure the request was sucessful
    if r["bboxes"]:
        print("[INFO] thread {} OK".format(n))
    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))

# loop over the number of threads
for i in range(0, cfg.getint('test', 'request_num')):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    sleep(cfg.getfloat('test', 'sleep'))

# insert a long sleep so we can wait until the server is finished
# processing the images
sleep(300)