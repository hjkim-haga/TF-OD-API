# PYTHONPATH=/home/ubuntu/myproj/models/ python crop_around_box.py -s original -d resized -w 1536 -h 1536

import argparse
import os
import xml.etree.ElementTree as ET

import albumentations as A
import cv2

from research.object_detection.custom_utils.annotations import read_pascal_voc


# pacal voc는 절대 좌표로 [x_min, y_min, x_max, y_max]를 표현하는데 반해,
# albumentations는 norm 된 [x_min, y_min, x_max, y_max]를 사용한다.

def normalize_bbox(width, height, xmin, ymin, xmax, ymax):
    """Normalize the bbox positions.
    Normalize the bbox postion by the related image size.
    
    :param width: image width
    :param height: image height
    :param xmin: absolute left x coordinate.
    :param ymin: absolute upper y coordinate.
    :param xmax: absolute right x coordinate.
    :param ymax: absolute lower y coordinate.
    :return: normalized (xmin, ymin, xmax, ymax)
    """
    return xmin / width, ymin / height, xmax / width, ymax / height


def unnormalize_bbox(width, height, xmin, ymin, xmax, ymax):
    """Normalize the bbox positions.
    Normalize the bbox postion by the related image size.

    :param width: image width
    :param height: image height
    :param xmin: relative left x coordinate.
    :param ymin: relative upper y coordinate.
    :param xmax: relative right x coordinate.
    :param ymax: relative lower y coordinate.
    :return: absolute (xmin, ymin, xmax, ymax)
    """
    return xmin * width, ymin * height, xmax * width, ymax * height


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


parser = argparse.ArgumentParser(description='args for cropping image with required resolution around bbox.',
                                 add_help=False)  # 없으면 -h 옵션 중복 에러.
parser.add_argument('--src', '-s', type=str, help='source directory containing images and annotations.')
parser.add_argument('--dest', '-d', type=str, help='destiny directory where images and annotations are saved.')
parser.add_argument('--width', '-w', type=int, help='required image width for your model.')
parser.add_argument('--height', '-h', type=int, help='required image height for your model.')
args = parser.parse_args()

# Required resolutions for detectors
width_efficientdet7 = height_efficientdet7 = 1536
width_spinenet190 = height_spinenet190 = 1536
width_efficientdet4 = height_efficientdet4 = 1024
width_spinenet143 = height_spinenet143 = 1280

if __name__ == '__main__':
    src = args.src
    dest = args.dest
    required_width = args.width
    required_height = args.height
    
    # Get the images and annotations
    images = (f for f in os.scandir(src) if os.path.splitext(f)[1] == '.jpg')
    annotations = (f for f in os.scandir(src) if os.path.splitext(f)[1] == '.xml')
    
    # Get transformer for cropping.
    transform = A.Compose([
        A.RandomSizedBBoxSafeCrop(width=required_width, height=required_height, always_apply=True),
    ], bbox_params=A.BboxParams(format='pascal_voc'))
    
    for ipath, apath in zip(images, annotations):
        img_filename = os.path.basename(ipath)
        annot_filename = os.path.basename(apath)
        print(img_filename)
        
        # Crop around the bbox
        i = cv2.imread(os.path.abspath(ipath))
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        adict = read_pascal_voc(apath)
        xmin, ymin, xmax, ymax = adict['xmin'], adict['ymin'], adict['xmax'], adict['ymax']
        b0 = [xmin, ymin, xmax, ymax, adict['name']]
        b = [b0]
        transformed = transform(image=i, bboxes=b)
        
        # Save the cropped image and related annotations.
        cv2.imwrite(os.path.join(dest, img_filename), transformed['image'])
        xmin, ymin, xmax, ymax, label = transformed['bboxes'][0]
        new_annot = {
            'width': required_width,
            'height': required_height,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }
        modify_pascal_voc(
            os.path.abspath(apath),
            os.path.join(dest, annot_filename),
            **new_annot)
