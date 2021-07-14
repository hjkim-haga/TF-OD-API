from PIL import Image, ExifTags
import argparse
from collections import namedtuple
import glob
import json
import os
from PIL import Image, ImageDraw


bbox_yolo = namedtuple('BboxYolo', ('cen_x', 'cen_y', 'width', 'height'))
bbox_haga = namedtuple('BboxHaga', ('x0', 'y0', 'x1', 'y1'))
img_info = namedtuple('Img', ('w', 'h'))


def inspect_bbox(bbox: namedtuple):
    """Inspect the validation of a bbox coordinates.

    :param bbox: it should hold x0, x1, y0, y1 in absolute positions.
    :return: results about out-of-image, upper-left < lower-right
    """

    result = []
    if bbox.x0 >= bbox.x1:
        result.append('x')
    if  bbox.y0 >= bbox.y1:
        result.append('y')
    
    if bbox.x1 > bbox.w:
        result.append('out-of-width')
    if bbox.y1 > bbox.h:
        result.append('out-of-height')
        
    return result


def rescale_bbox(annot: namedtuple) -> namedtuple:
    """Return rescaled coordinates of bbox in the image size. 

    YOLO requires the bounding box coordinates should be 
    (relative_center_x, relative_center_y, relative_w, relative_h)
    
    :param namedtuple annot: holds each vertex absoulte positions.
    :return namedtuple: rescaled bbox
    """
    x0 = annot.x0
    y0 = annot.y0
    x1 = annot.x1
    y1 = annot.y1
    img_w = annot.img_w
    img_h = annot.img_h

    bbox_w = abs(x1 - x0)
    bbox_h = abs(y1 - y0)

    cen_x = x0 + bbox_w/2
    cen_y = y0 + bbox_h/2

    rel_cen_x = cen_x / img_w
    rel_cen_y = cen_y / img_h

    rel_w = bbox_w / img_w
    rel_h = bbox_h / img_h

    rescaled_bbox = namedtuple('Bbox', ['rel_cen_x', 'rel_cen_y', 'rel_w', 'rel_h'])
    return rescaled_bbox(rel_cen_x, rel_cen_y, rel_w, rel_h)


def get_exif(path):
    img = Image.open(path)
    exif = img._getexif()
    if not exif:
        result = None
    else:
        result = exif
    return result


def calibrate_bbox(annotation: namedtuple) -> namedtuple:
    """Calibrate the wrong coordinates
    :param annotation: contains bbox coordinates and img size
    """
    
    x0 = annotation.bbox_x0
    x1 = annotation.bbox_x1
    y0 = annotation.bbox_y0
    y1 = annotation.bbox_y1
    w = annotation.img_w
    h = annotation.img_h

    x0 = 1 if x0 <= 0 else x0
    y0 = 1 if y0 <= 0 else y0
    x1 = w - 1 if x1 >= w else x1
    y1 = h - 1 if y1 >= h else y1


def read_annotations(path: str) -> namedtuple:
    """ Return the bbox Coordinates.

    This function does 'calibrate' the coordinates than just returning them.
    For example, if the image height is '2048', but x1 is '4089'.
    then this function automatically set x1 smaller than height.

    :param str path: the path to annotation file in JSON format
    :return: contains category, bbox coordinates.
    """
    with open(path) as fin:
        annot = json.load(fin)
    x0 = float(annot['shapes']['points'][0][0])
    y0 = float(annot['shapes']['points'][0][1])
    x1 = float(annot['shapes']['points'][1][0])
    y1 = float(annot['shapes']['points'][1][1])
    w = float(annot['imageWidth'])
    h = float(annot['imageHeight'])
    cat = annot['shapes']['label']

    # Calibrated pos
    annot = namedtuple('annot', ['cat', 'img_w', 'img_h', 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'])
    return annot(cat, w, h, x0, y0, x1, y1)


def read_yolo_annot(annot_path: str):
    with open(annot_path, 'r') as fin:
        annot = fin.readline().split(' ')  # [0] is class id.
    return bbox_yolo(float(annot[1]), float(annot[2]), float(annot[3]), float(annot[4]))


def adjust_yolo_annot(img_info: namedtuple, bbox_yolo: namedtuple) -> namedtuple:
    """ Adjust the YOLO annotation to CV2.

    :param namedtuple img_info: image width & height
    :param namedtuple bbox_yolo: bbox center x & y, w & h
    :return result: x0, y0, x1, y1
    """
    img_width = img_info.w
    img_height = img_info.h

    cen_x = bbox_yolo.cen_x * img_width
    cen_y = bbox_yolo.cen_y * img_height
    w = bbox_yolo.width * img_width
    h = bbox_yolo.height * img_height

    x0 = cen_x - w/2
    x1 = cen_x + w/2
    y0 = cen_y - h/2
    y1 = cen_y + h/2

    return bbox_haga(x0, y0, x1, y1)