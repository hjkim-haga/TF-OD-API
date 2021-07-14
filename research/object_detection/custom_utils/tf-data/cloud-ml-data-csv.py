import argparse
import json
import os
import sys
sys.path.append('/mnt/d/my-scripts/')

from annotations.modules import read_annotations
from general.module import filter_by_type


def make_tflite_csv(directory: str, out_file_name: str):
    """Make CSV file for TensorFlow Lite Model Maker.

    a format of one line is like:
        `TRAINING,dataset/train/image1.png,bike,0.7,0.6,,,0.8,0.9,,`
    :param directory: 3 types: {train/validation/test}.
    :param out_file: the CSV file name without extension.
    """
    fout = open(out_file_name + '.csv', 'a')

    file_list = filter_by_type(directory, '.jpg')
    if (dataset := os.path.basename(directory)) == 'train':
        dataset = 'TRAINING'
    elif dataset == 'validation':
        dataset = 'VALIDATION'
    elif dataset == 'test':
        dataset = 'TEST'

    for f in file_list:
        annot = read_annotations(f)
        full_path = os.path.abspath(f)
        # Google Cloud requires relative positions.
        x0 = annot.bbox_x0 / annot.img_w
        x1 = annot.bbox_x1 / annot.img_w
        y0 = annot.bbox_y0 / annot.img_h
        y1 = annot.bbox_y1 / annot.img_h

        line = f'{dataset.upper()},{full_path},{annot.cat},{x0},{y0},' + \
                f',,{x1},{y1},,\n'
        fout.write(line)
    fout.close()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--root', type=str, required=True, help='Dataset root directory.')
args = parser.parse_args()


if __name__ == '__main__':
    root_dir = args.root
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'validation')
    test_dir = os.path.join(root_dir, 'test')

    make_tflite_csv(train_dir, 'electronic_ml_use')
    make_tflite_csv(valid_dir, 'electronic_ml_use')
    make_tflite_csv(test_dir, 'electronic_ml_use')