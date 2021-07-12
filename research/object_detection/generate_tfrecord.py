"""
Usage:
  # From tensorflow/models/
  train: 160장, valid: 40장, test: 20장
  # Create train data:
  PYTHONPATH=$PYTHONPATH:/home/ubuntu/myproj/models/:/home/ubuntu/myproj/models/research:/home/ubuntu/myproj/models/research/object_detection python generate_tfrecord.py \
    --csv_input=/home/ubuntu/haga-dataset/electronics/train_labels.csv \
    --image_dir=/home/ubuntu/haga-dataset/electronics/train \
    --class_names=/home/ubuntu/haga-dataset/electronics/class-names.txt \
    --output_path=/home/ubuntu/haga-dataset/electronics/train.record \
    --num_shards=200
  
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import os
from collections import namedtuple

import contextlib2
import pandas as pd
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

from utils import dataset_util

# All path should be absolute.
flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input.')
flags.DEFINE_string('image_dir', '', 'Path to the image directory.')
flags.DEFINE_string('class_names', '', 'Path to the file containing class names.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord.')
flags.DEFINE_string('num_shards', '', 'Number of record files.')
FLAGS = flags.FLAGS

file = open(FLAGS.class_names, 'r')
output_dict = {}  # 각 클래스에 id 번호 부여.
classname = file.readline().strip()
count = 1
while len(classname) > 0:
    output_dict[classname] = count
    classname = file.readline().strip()
    count += 1
file.close()


# TO-DO(sombody): replace this with label map
def class_text_to_int(row_label):
    return output_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, f'{group.filename}'), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_filepath,
                     num_shards,
                     image_dir,
                     grouped):
    """Creates a TFRecord file from examples.
  
    Args:
      output_filepath: Path to where output file is saved.
      num_shards: Number of shards for output file.
      image_dir: Directory where image files are stored.
      grouped: `split(examples, 'filename')`.
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filepath, num_shards)
        
        for idx, group in enumerate(grouped):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(grouped))
            
            tf_example = create_tf_example(group, image_dir)
            if tf_example:
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())


def main(_):
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    _, record_name = os.path.split(FLAGS.output_path)
    create_tf_record(FLAGS.output_path, int(FLAGS.num_shards), FLAGS.image_dir, grouped)


if __name__ == '__main__':
    tf.app.run()
