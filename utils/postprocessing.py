"""
This script provide postprocessing for inference
Copyright 2022 Huawei Technologies Co., Ltd

CREATED:  2020-6-04 20:12:13
MODIFIED: 2021-02-05 23:48:45
"""
# -*- coding:utf-8 -*-
import numpy as np

from time import time
from data.constant import ACL_MEMCPY_DEVICE_TO_HOST


def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float)

def _sigmoid(x0):
    s = 1 / (1 + np.exp(-x0))
    return s

def detect(x, c, model_type="yolov5"):
    """
    x(bs,3,20,20,85)
    """
    # x(bs,3,20,20,85)
    z = []
    grid = []
    for i in range(3):
        _, _, ny, nx, _ =  x[i].shape
        grid.append(_make_grid(nx, ny))
    if model_type == 'yolov5':
        stride =  np.array([8, 16, 32])
        anchor_grid = np.array(
            [[10., 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])\
            .reshape(3, 1, 3, 1, 1, 2)
    elif model_type == 'yolov3':
        stride = np.array([32, 16, 8])
        anchor_grid = np.array(
            [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10., 13, 16, 30, 33, 23]])\
            .reshape(3, 1, 3, 1, 1, 2)

    for i in range(3):
        y = _sigmoid(x[i])
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.reshape(1, -1, c))
    return np.concatenate(z, 1)


def _xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    return y

def _nms(boxes, scores, iou_thres):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.zeros(1)
    
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float32")
    
    # initialize the list of picked indexes	
    pick = []
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_thres)[0])))

    # return only picked value
    return np.array(pick)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after

    t = time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = _xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        i, j = np.nonzero(x[:, 5:] > conf_thres)
        x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype('float32')), 1)

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = _nms(boxes, scores, iou_thres)
       
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        if (time() - t) > time_limit:
            break  # time limit exceeded
        
    return output


def _clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    np.clip(boxes[:, 0], 0, img_shape[1])
    np.clip(boxes[:, 1], 0, img_shape[0])
    np.clip(boxes[:, 2], 0, img_shape[1])
    np.clip(boxes[:, 3], 0, img_shape[0])

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    _clip_coords(coords, img0_shape)

    return coords