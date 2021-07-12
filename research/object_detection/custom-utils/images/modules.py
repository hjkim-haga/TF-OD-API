import json
import os
from collections import namedtuple
from PIL import Image, ImageDraw
import numpy as np

def draw_bbox(img_path: str, bbox: namedtuple, line_width=10):
    """ Draw a bbox around the object.

    :param str image_path: an image containing an object
    :param namedtuple bbox: x0, y0, x1, y1
    :param int line_width: a width of bbox side
    :return draw: image object with bbox drawed.
    """

    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    draw.rectangle(((bbox.x0, bbox.y0), (bbox.x1, bbox.y1)), outline="red", width=line_width)

    return im


def resize_image(img, height:int, width:int):
    return img.resize((width, height))


def crop_image(img, region:namedtuple) -> np.ndarray:
    """Crop an image.

    :param img: Assume that the image is read by OpenCV.imread
    :param region: ROI with (x0, y0, x1, y1)
    """
    
    x0, y0, x1, y1 = region
    return img.crop((x0, y0, x1, y1))


def crop_and_resize_image(img:np.ndarray, region:namedtuple, height:int, width:int):
    return resize_image(crop_image(img, region), height, width)


def collect_only_images(dir:str):
    only_images = []
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[1]
        if  ext == '.jpg':
            only_images.append(file)
    return only_images


def resize_img_holding_bbox(imgpath, destpath=''):
    """
        Args:
            imgpath (str): 
            destpath (str): 지정되지 않았다면 원본 이미지를 덮어쓴다
    """
    basename = os.path.splitext(os.path.basename(imgpath))[0]
    jsfilename = basename + ".json"

    # Load bbox info
    with open(os.path.join(os.path.dirname(imgpath), jsfilename), 'r') as jsfile:
        jsobj = json.load(jsfile)

        img_height = jsobj["imageHeight"]
        img_width  = jsobj["imageWidth"]
        img_area   = img_height * img_width
        bbox_x0    = jsobj["shapes"]["points"][0][0]  # (x0, y0)
        bbox_y0    = jsobj["shapes"]["points"][0][1]  # (x0, y0)
        bbox_x1    = jsobj["shapes"]["points"][1][0]  # (x1, y1)
        bbox_y1    = jsobj["shapes"]["points"][1][1]  # (x1, y1)
        bbox_w     = bbox_x1 - bbox_x0
        bbox_h     = bbox_y1 - bbox_y0


    threshold_are = 8000000
    if img_area < threshold_are:
        print("No need to resize {}. Skip.".format(basename+".jpg"))
        return
    print("Resizing {}".format(basename+".jpg"))

    scale = .5
    re_img_width = int(img_width * scale)
    re_img_height = int(img_height * scale)
    re_bbox_w = int(bbox_w * scale)
    re_bbox_h = int(bbox_h * scale)
    re_bbox_x0 = int(bbox_x0 * scale)
    re_bbox_y0 = int(bbox_y0 * scale)
    re_bbox_x1 = int(re_bbox_x0 + re_bbox_w)
    re_bbox_y1 = int(re_bbox_y0 + re_bbox_h)

    # Rewrite the json file
    jsobj["imageHeight"] = re_img_height
    jsobj["imageWidth"] = re_img_width
    jsobj["shapes"]["points"][0][0] = re_bbox_x0
    jsobj["shapes"]["points"][0][1] = re_bbox_y0
    jsobj["shapes"]["points"][1][0] = re_bbox_x1
    jsobj["shapes"]["points"][1][1] = re_bbox_y1
    with open(os.path.join(os.path.dirname(imgpath), jsfilename), 'w') as jsfile:
        json.dump(jsobj, jsfile, indent=4)
        

    # Resize the image
    with Image.open(imgpath) as im:

        # im.show()
        # draw = ImageDraw.Draw(im)
        # draw.rectangle(((bbox_x0, bbox_y0), (bbox_x1, bbox_y1)), outline="blue", width=10)
        # im.save(os.path.join(os.path.dirname(args['dest']), basename + "-or.jpg"), "jpeg")

        resize_im = im.resize((re_img_width, re_img_height))
        # draw = ImageDraw.Draw(resize_im)
        # draw.rectangle(((re_bbox_x0, re_bbox_y0), (re_bbox_x1, re_bbox_y1)), outline="red", width=10)
        
        # resize_im.show()
        resize_im.save(os.path.join(destpath), "jpeg")

        