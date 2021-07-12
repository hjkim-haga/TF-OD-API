# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the voucher dataset to TFRecord for object_detection.

mv bed/bed_000{1201..1300}.* testdev/
mv doorlock/doorlock_000{1201..1300}.* testdev/
mv drawer/drawer_000{1201..1300}.* testdev/
mv knob/knob_000{1201..1300}.* testdev/
mv multitap/multitap_000{1201..1300}.* testdev/
mv socket/socket_000{1201..1300}.* testdev/
mv sofa/sofa_000{1201..1300}.* testdev/

mv bed/bed_000{1001..1200}.* valid/
mv doorlock/doorlock_000{1001..1200}.* valid/
mv drawer/drawer_000{1001..1200}.* valid/
mv knob/knob_000{1001..1200}.* valid/
mv multitap/multitap_000{1001..1200}.* valid/
mv socket/socket_000{1001..1200}.* valid/
mv sofa/sofa_000{1001..1200}.* valid/

mv bed/bed_*.* train/
mv doorlock/doorlock_*.* train/
mv drawer/drawer_*.* train/
mv knob/knob_*.* train/
mv multitap/multitap_*.* train/
mv socket/socket_*.* train/
mv sofa/sofa_*.* train/


Directory structure:
    scissors/  <-- this becomes 'data_dir/'.
    |-- images/
        |-- train/
        |-- valid/
        |-- testdev/
    |-- annotations/
        |-- train/  <-- files whose name is the same with the corresponding images. Its format follows voucher's.
        |-- valid/
        |-- testdev/
        |
        |-- train.txt  # <-- line-format '<img_filename_without_extension>'
        |-- valid.txt  # To make list file, use 'object_detection/dataset_tools/list_examples.jl'
        |-- testdev.txt
Example usage:

    python object_detection/dataset_tools/create_voucher_tf_record.py \
        --data_dir=/home/jun/models/research/object_detection/data/custom/ \
        --output_dir=/home/jun/models/research/object_detection/data/custom/ \
        --label_map_path=/home/jun/models/research/object_detection/data/custom_label_map.pbtxt
"""   

import json
import hashlib
import io
import logging
import os
import random
import re

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset. It contains "images/" and "annotations/".')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
flags.DEFINE_boolean('bbox_on', True, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
flags.DEFINE_integer('num_train_shards', 8, 'Number of TFRecord shards')
flags.DEFINE_integer('num_valid_shards', 0, 'Number of TFRecord shards')
flags.DEFINE_integer('num_testdev_shards', 0, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

# Only used in dict_to_tf_example()
def get_class_name_from_filename(file_name):
  """Gets the class name from a file obeying the voucher naming. Complete!

  Args:
    file_name: The file name to get the class name from.
               ie. "bed_0000001.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       bbox_on=True):
  """Convert voucher-json derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding voucher JSON fields for a single image (parsed by lib 'json')
    label_map_dict: A map from string label names to integers ids. A specific form follows as:
      item {
        id: 1
        name: 'bed'
      }
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data. ex) images/train/
    bbox_on: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['imagePath'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['imageWidth'])
  height = int(data['imageHeight'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []

  xmin = float(data['shapes']['points'][0][0])
  ymin = float(data['shapes']['points'][0][1])
  xmax = float(data['shapes']['points'][1][0])
  ymax = float(data['shapes']['points'][1][1])

  # Normalize the points.
  xmins.append(xmin / width)
  ymins.append(ymin / height)
  xmaxs.append(xmax / width)
  ymaxs.append(ymax / height)

  class_name = data['shapes']['label']
  classes_text.append(class_name.encode('utf8'))
  classes.append(label_map_dict[class_name])

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['imagePath'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['imagePath'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }
  

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,
                     bbox_on=True):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    bbox_on: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    # `output_tfrecords` holds the list of tf_example created.
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):  # 'examples' is a target file list without extensions.
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      json_path = os.path.join(annotations_dir, example + '.json')

      if not os.path.exists(json_path):
        logging.warning('Could not find %s, ignoring example.', json_path)
        continue
      with open(json_path, 'r') as fid:
        data = json.load(fid)

      try:
        tf_example = dict_to_tf_example(
            data,
            label_map_dict,
            image_dir,
            bbox_on=bbox_on)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', json_path)


def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  # I made this list becaue I don't know how to abstract num_shard by dataset.
  datasets = ['train', 'valid', 'testdev']
  shard_by_ds = [FLAGS.num_train_shards, FLAGS.num_valid_shards, FLAGS.num_testdev_shards]

  for i, ds_type in enumerate(datasets):
    num_shards = shard_by_ds[i]
    if num_shards <= 0:
      logging.info(f'{ds_type} dataset skipped.')
      continue
    image_dir = os.path.join(data_dir, 'images', ds_type)
    annotation_dir = os.path.join(data_dir, 'annotations', ds_type)
    examples_path = os.path.join(data_dir, 'annotations', ds_type + '.txt')
    examples = dataset_util.read_examples_list(examples_path)
    output_path = os.path.join(FLAGS.output_dir, 'voucher_' + ds_type + '.record')

    
    logging.info(f'Reading from {ds_type} dataset')
    create_tf_record(
        output_path,
        num_shards,
        label_map_dict,
        annotation_dir,
        image_dir,
        examples,
        bbox_on=FLAGS.bbox_on)

    logging.info(f'{ds_type}: {examples}')


if __name__ == '__main__':
  tf.app.run()
