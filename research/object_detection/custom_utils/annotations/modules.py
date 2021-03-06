import collections
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple

from PIL import Image

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
    if bbox.y0 >= bbox.y1:
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
    
    cen_x = x0 + bbox_w / 2
    cen_y = y0 + bbox_h / 2
    
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
    w =  annotation.img_w
    h =  annotation.img_h
    
    x0 = 1 if x0 <= 0 else x0
    y0 = 1 if y0 <= 0 else y0
    x1 = w - 1 if x1 >= w else x1
    y1 = h - 1 if y1 >= h else y1
    
    return x0, y0, x1, y1


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
    
    x0 = cen_x - w / 2
    x1 = cen_x + w / 2
    y0 = cen_y - h / 2
    y1 = cen_y + h / 2
    
    return bbox_haga(x0, y0, x1, y1)


def read_pascal_voc(path: str) -> dict:
    # Parse the xml
    tree = ET.parse(os.path.abspath(path))
    root = tree.getroot()
    size_tag = root.find('size')
    object_tag = root.find('object')
    bndbox_tag = object_tag.find('bndbox')
    
    # Get the annotation
    width = int(size_tag.find('width').text)
    height = int(size_tag.find('height').text)
    name = object_tag.find('name').text
    xmin = int(bndbox_tag.find('xmin').text)
    ymin = int(bndbox_tag.find('ymin').text)
    xmax = int(bndbox_tag.find('xmax').text)
    ymax = int(bndbox_tag.find('ymax').text)
    
    return {  # bbox not normalized
        'filename': os.path.basename(path),
        'name': name,
        'width': width,
        'height': height,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
    }


def modify_pascal_voc(path, new_path, **kwargs):
    """Modify pascal voc xml and save it

    :param path: path to pascal voc will be modified.
    :param new_path: path to modified file.
    :param dict kwargs: key is the elem name, value is the new value for the element.
    :return: None
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    all_desc = []
    for elem, val in kwargs.items():
        elem_iter = root.iter(elem)
        for el in elem_iter:
            all_desc.append(el)
    
    for el in all_desc:
        el.text = str(kwargs[el.tag])
    
    tree.write(new_path)


class Coco:
    def __init__(self, file_path):
        self.coco = self.read_json(file_path)
    
    def read_json(self, file_path):
        with open(file_path, 'r') as fin:
            coco = json.load(fin)
        return coco
    
    def find_by_file(self, file_path):
        """??????????????? coco ????????? ?????????.
        
        'images'?????? 'file_name'?????? ??????.
        
        :param file_name:
        :return: ???????????? 'images' ?????????, image_id ???????????? 'annotations' ?????????
        """
        file_name = os.path.split(os.path.abspath(file_path))[-1]
        images = []
        annotations = []
        image_id = None
        # Find file id
        for image_obj in self.coco['images']:
            if file_name == image_obj['file_name']:
                images.append(image_obj)
                image_id = image_obj['id']
        
        for annot_obj in self.coco['annotations']:
            if image_id == annot_obj['image_id']:
                annotations.append(annot_obj)
        
        return images, annotations
    
    def lighten_coco(self, directory):
        """?????? ???????????? ??? ????????? ?????? ???????????? ?????? coco ?????? ?????? ??????
        
        :param directory: ?????? ??????????????? ?????? ?????? ???????????? ??????
        :return: directory??? ?????? ??????????????? annotation??? ?????? ?????? ?????? ??????
        """
        light_coco = {
            'images': [],
            'annotations': [],
            'categories': [
                {
                    "supercategory": "person",
                    "id": 1,
                    "name": "person"
                },
                {
                    "supercategory": "vehicle",
                    "id": 2,
                    "name": "bicycle"
                },
                {
                    "supercategory": "vehicle",
                    "id": 3,
                    "name": "car"
                },
                {
                    "supercategory": "vehicle",
                    "id": 4,
                    "name": "motorcycle"
                },
                {
                    "supercategory": "vehicle",
                    "id": 5,
                    "name": "airplane"
                },
                {
                    "supercategory": "vehicle",
                    "id": 6,
                    "name": "bus"
                },
                {
                    "supercategory": "vehicle",
                    "id": 7,
                    "name": "train"
                },
                {
                    "supercategory": "vehicle",
                    "id": 8,
                    "name": "truck"
                },
                {
                    "supercategory": "vehicle",
                    "id": 9,
                    "name": "boat"
                },
                {
                    "supercategory": "outdoor",
                    "id": 10,
                    "name": "traffic light"
                },
                {
                    "supercategory": "outdoor",
                    "id": 11,
                    "name": "fire hydrant"
                },
                {
                    "supercategory": "outdoor",
                    "id": 13,
                    "name": "stop sign"
                },
                {
                    "supercategory": "outdoor",
                    "id": 14,
                    "name": "parking meter"
                },
                {
                    "supercategory": "outdoor",
                    "id": 15,
                    "name": "bench"
                },
                {
                    "supercategory": "animal",
                    "id": 16,
                    "name": "bird"
                },
                {
                    "supercategory": "animal",
                    "id": 17,
                    "name": "cat"
                },
                {
                    "supercategory": "animal",
                    "id": 18,
                    "name": "dog"
                },
                {
                    "supercategory": "animal",
                    "id": 19,
                    "name": "horse"
                },
                {
                    "supercategory": "animal",
                    "id": 20,
                    "name": "sheep"
                },
                {
                    "supercategory": "animal",
                    "id": 21,
                    "name": "cow"
                },
                {
                    "supercategory": "animal",
                    "id": 22,
                    "name": "elephant"
                },
                {
                    "supercategory": "animal",
                    "id": 23,
                    "name": "bear"
                },
                {
                    "supercategory": "animal",
                    "id": 24,
                    "name": "zebra"
                },
                {
                    "supercategory": "animal",
                    "id": 25,
                    "name": "giraffe"
                },
                {
                    "supercategory": "accessory",
                    "id": 27,
                    "name": "backpack"
                },
                {
                    "supercategory": "accessory",
                    "id": 28,
                    "name": "umbrella"
                },
                {
                    "supercategory": "accessory",
                    "id": 31,
                    "name": "handbag"
                },
                {
                    "supercategory": "accessory",
                    "id": 32,
                    "name": "tie"
                },
                {
                    "supercategory": "accessory",
                    "id": 33,
                    "name": "suitcase"
                },
                {
                    "supercategory": "sports",
                    "id": 34,
                    "name": "frisbee"
                },
                {
                    "supercategory": "sports",
                    "id": 35,
                    "name": "skis"
                },
                {
                    "supercategory": "sports",
                    "id": 36,
                    "name": "snowboard"
                },
                {
                    "supercategory": "sports",
                    "id": 37,
                    "name": "sports ball"
                },
                {
                    "supercategory": "sports",
                    "id": 38,
                    "name": "kite"
                },
                {
                    "supercategory": "sports",
                    "id": 39,
                    "name": "baseball bat"
                },
                {
                    "supercategory": "sports",
                    "id": 40,
                    "name": "baseball glove"
                },
                {
                    "supercategory": "sports",
                    "id": 41,
                    "name": "skateboard"
                },
                {
                    "supercategory": "sports",
                    "id": 42,
                    "name": "surfboard"
                },
                {
                    "supercategory": "sports",
                    "id": 43,
                    "name": "tennis racket"
                },
                {
                    "supercategory": "kitchen",
                    "id": 44,
                    "name": "bottle"
                },
                {
                    "supercategory": "kitchen",
                    "id": 46,
                    "name": "wine glass"
                },
                {
                    "supercategory": "kitchen",
                    "id": 47,
                    "name": "cup"
                },
                {
                    "supercategory": "kitchen",
                    "id": 48,
                    "name": "fork"
                },
                {
                    "supercategory": "kitchen",
                    "id": 49,
                    "name": "knife"
                },
                {
                    "supercategory": "kitchen",
                    "id": 50,
                    "name": "spoon"
                },
                {
                    "supercategory": "kitchen",
                    "id": 51,
                    "name": "bowl"
                },
                {
                    "supercategory": "food",
                    "id": 52,
                    "name": "banana"
                },
                {
                    "supercategory": "food",
                    "id": 53,
                    "name": "apple"
                },
                {
                    "supercategory": "food",
                    "id": 54,
                    "name": "sandwich"
                },
                {
                    "supercategory": "food",
                    "id": 55,
                    "name": "orange"
                },
                {
                    "supercategory": "food",
                    "id": 56,
                    "name": "broccoli"
                },
                {
                    "supercategory": "food",
                    "id": 57,
                    "name": "carrot"
                },
                {
                    "supercategory": "food",
                    "id": 58,
                    "name": "hot dog"
                },
                {
                    "supercategory": "food",
                    "id": 59,
                    "name": "pizza"
                },
                {
                    "supercategory": "food",
                    "id": 60,
                    "name": "donut"
                },
                {
                    "supercategory": "food",
                    "id": 61,
                    "name": "cake"
                },
                {
                    "supercategory": "furniture",
                    "id": 62,
                    "name": "chair"
                },
                {
                    "supercategory": "furniture",
                    "id": 63,
                    "name": "couch"
                },
                {
                    "supercategory": "furniture",
                    "id": 64,
                    "name": "potted plant"
                },
                {
                    "supercategory": "furniture",
                    "id": 65,
                    "name": "bed"
                },
                {
                    "supercategory": "furniture",
                    "id": 67,
                    "name": "dining table"
                },
                {
                    "supercategory": "furniture",
                    "id": 70,
                    "name": "toilet"
                },
                {
                    "supercategory": "electronic",
                    "id": 72,
                    "name": "tv"
                },
                {
                    "supercategory": "electronic",
                    "id": 73,
                    "name": "laptop"
                },
                {
                    "supercategory": "electronic",
                    "id": 74,
                    "name": "mouse"
                },
                {
                    "supercategory": "electronic",
                    "id": 75,
                    "name": "remote"
                },
                {
                    "supercategory": "electronic",
                    "id": 76,
                    "name": "keyboard"
                },
                {
                    "supercategory": "electronic",
                    "id": 77,
                    "name": "cell phone"
                },
                {
                    "supercategory": "appliance",
                    "id": 78,
                    "name": "microwave"
                },
                {
                    "supercategory": "appliance",
                    "id": 79,
                    "name": "oven"
                },
                {
                    "supercategory": "appliance",
                    "id": 80,
                    "name": "toaster"
                },
                {
                    "supercategory": "appliance",
                    "id": 81,
                    "name": "sink"
                },
                {
                    "supercategory": "appliance",
                    "id": 82,
                    "name": "refrigerator"
                },
                {
                    "supercategory": "indoor",
                    "id": 84,
                    "name": "book"
                },
                {
                    "supercategory": "indoor",
                    "id": 85,
                    "name": "clock"
                },
                {
                    "supercategory": "indoor",
                    "id": 86,
                    "name": "vase"
                },
                {
                    "supercategory": "indoor",
                    "id": 87,
                    "name": "scissors"
                },
                {
                    "supercategory": "indoor",
                    "id": 88,
                    "name": "teddy bear"
                },
                {
                    "supercategory": "indoor",
                    "id": 89,
                    "name": "hair drier"
                },
                {
                    "supercategory": "indoor",
                    "id": 90,
                    "name": "toothbrush"
                }
            ],
        }
        for file in os.scandir(directory):
            if os.path.splitext(file)[1] != '.jpg':
                print(f'cannot read {file}')
                sys.exit(1)
            images_found, annotations_found = self.find_by_file(file)
            light_coco['images'].extend(images_found)
            light_coco['annotations'].extend(annotations_found)
        
        return light_coco


if __name__ == '__main__':
    dataset_path = '/home/ubuntu/haga-dataset/electronics/train/'
    annot_list = (f for f in os.scandir(dataset_path) if os.path.splitext(f)[1] == '.xml')
    for file in annot_list:
        pascal_annot = read_pascal_voc(file)
        Annot = namedtuple('Annot', ('bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'img_w', 'img_h'))
        annot = Annot(bbox_x0=pascal_annot['xmin'],
                      bbox_y0=pascal_annot['ymin'],
                      bbox_x1=pascal_annot['xmax'],
                      bbox_y1=pascal_annot['ymax'],
                      img_w=pascal_annot['width'],
                      img_h=pascal_annot['height'])
        calib_bbox = calibrate_bbox(annot)  # x0, y0, x1, y1
        modify_pascal_voc(os.path.abspath(file), os.path.abspath(file),
                          **{'xmin': calib_bbox[0],
                             'ymin': calib_bbox[1],
                             'xmax': calib_bbox[2],
                             'ymax': calib_bbox[3]})
        print(f'{os.path.basename(file)} done.')

    
    # # ??? ???????????? ????????? ?????? ?????? annotations ??? ??????
    # test_dir = 'test_images/ensemble-test'
    # result_path = 'test_images/light_coco_train.json'
    #
    # # # train
    # train_file = 'test_images/test/light_coco.json'
    # light_coco_train = Coco(train_file)
    #
    # image_ids = set()
    # for image_obj in light_coco_train.coco['images']:
    #     image_ids.add(image_obj['id'])
    # d = collections.defaultdict()
    # for i in image_ids:
    #     d[i] = 0
    # image_ids = d
    #
    # for annot_obj in light_coco_train.coco['annotations']:
    #     image_ids[annot_obj['image_id']] += 1
    # max_num_annotations_per_img = 0
    # for num_annot in image_ids.values():
    #     if max_num_annotations_per_img < num_annot:
    #         max_num_annotations_per_img = num_annot
    # print(max_num_annotations_per_img)
    
    # # ????????? ?????? ?????? ?????? ??????????????? ?????? ?????? ?????? ?????????
    # with open(result_path, 'w') as fout:
    #     json.dump(lighten_coco_train, fout)
    #
    # # valid
    # result_path = 'test_images/light_coco_val.json'
    # val_file = 'test_images/instances_val2017.json'
    # coco_val = Coco(val_file)
    # lighten_coco_val = coco_val.lighten_coco(test_dir)
    # print(len(lighten_coco_val['images']))
    # with open(result_path, 'w') as fout:
    #     json.dump(lighten_coco_val, fout)
