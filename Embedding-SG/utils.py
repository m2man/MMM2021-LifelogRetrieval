import networkx as nx
from PIL import Image
import numpy as np
import cv2

def visualize_Image(pil_img, G):
    extend_row = 20
    sample_img_np = np.asarray(pil_img)
    original_size = sample_img_np.shape
    black_extend = 255*np.ones((extend_row, original_size[1], 3), dtype='uint8')
    
    sample_img_np = np.vstack((black_extend, sample_img_np))
    nodes_info = list(G.nodes(data=True))
    for node in nodes_info:
        label = node[1]['decode_obj'] # + ':' + node[0].split(':')[-1]
        box = node[1]['bbox']
        box = [int(x) for x in box]
        cv2.rectangle(sample_img_np, (box[0], box[1]+extend_row), (box[2], box[3]+extend_row), (0, 255, 0), 2)
        cv2.putText(sample_img_np, label, (box[0], box[1]+extend_row-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1)
        
    temp = Image.fromarray(sample_img_np)
    return temp

def visualize_Image_only_bbox(pil_img, bbox):
    extend_row = 20
    sample_img_np = np.asarray(pil_img)
    original_size = sample_img_np.shape
    black_extend = 255*np.ones((extend_row, original_size[1], 3), dtype='uint8')
    
    sample_img_np = np.vstack((black_extend, sample_img_np))
    
    for idx, box in enumerate(bbox):
        box = [int(x) for x in box]
        cv2.rectangle(sample_img_np, (box[0], box[1]+extend_row), (box[2], box[3]+extend_row), (0, 255, 0), 2)
        cv2.putText(sample_img_np, str(idx), (box[0], box[1]+extend_row-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1)
        
    temp = Image.fromarray(sample_img_np)
    return temp


def get_iou_v0(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Returns: float in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def percent_overlap(box1, box2):
    '''
    Calculate the percentage of area of box1 is overlap with box 2. 
    This is not IoU, and the order of parameter box1, box2 is important
    box = [x1, y1, x2, y2] (top-left, bottom-right)
    '''
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    percent = intersection_area / box1_area
    
    assert percent >= 0.0
    assert percent <= 1.0
    
    return percent