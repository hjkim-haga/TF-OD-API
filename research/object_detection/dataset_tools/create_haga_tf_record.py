r"""Convert raw HAGA dataset in the format PASCAL to TFRecord for object_detection.

Example usage:
    PYTHONPATH=$PYTHONPATH:/home/ubuntu/myproj/models/:/home/ubuntu/myproj/models/research:/home/ubuntu/myproj/models/research/object_detection python dataset_tools/create_haga_tf_record.py \
        --set train \
        --data_dir /home/ubuntu/haga-dataset/electronics/train_resize \
        --annotations_dir xmls \
        --images_dir jpegs \
        --output_path /home/ubuntu/haga-dataset/electronics/records/train.record \
        --num_shards 100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

parser = argparse.ArgumentParser(description='args for converting dataset to tfrecords.')
parser.add_argument('--data_dir', type=str, required=True, help='Root directory to raw PASCAL VOC dataset.')
parser.add_argument('--set', default='train', required=True, help='Convert training set, validation set or '
                    'merged set.')
parser.add_argument('--annotations_dir', default='xmls',
                    help='(Relative) path to annotations directory.')
parser.add_argument('--images_dir', default='jpegs',
                    help='(Relative) path to images directory.')
parser.add_argument('--output_path', default='', help='Path to output TFRecord')
parser.add_argument('--label_map_path', default='data/voucher_label_map.pbtxt',
                    help='Path to label map proto')
parser.add_argument('--ignore_difficult_instances', type=bool, default=False, help='Whether to ignore '
                     'difficult instances')
parser.add_argument('--num_shards', type=int, default=100, help='Number of shards for output file.')
args = parser.parse_args()


SETS = ['train', 'valid', 'test']


def list_examples(annot_dir: str, outfile: str) -> list:
    examples_list = sorted([os.path.splitext(f)[0] for f in os.listdir(annot_dir)])
    with open(outfile, 'w') as fout:
        for elem in examples_list:
            fout.write(elem  + '\n')

    
def dict_to_tf_example(data,
                       image_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_directory: String specifying directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  # Read an image
  img_path = os.path.join(image_directory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  width = int(data['size']['width'])
  height = int(data['size']['height'])
  
  # Get object annotations in the read image.
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
  
  # Make annotations into the example.
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main():
  if args.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  data_dir = args.data_dir
  if not os.path.exists(os.path.dirname(args.output_path)):
      os.mkdir(os.path.dirname(args.output_path))
  writers = [
      tf.io.TFRecordWriter(args.output_path + '-%05d-of-%05d.tfrecord' %
                           (i, args.num_shards))
      for i in range(args.num_shards)
  ]
  label_map_dict = label_map_util.get_label_map_dict(args.label_map_path)
  print(label_map_dict)
  annotations_dir = os.path.join(data_dir, args.annotations_dir)
  images_dir = os.path.join(data_dir, args.images_dir)
  examples_path = os.path.join(data_dir, args.set + '.txt')
  list_examples(annotations_dir, examples_path)  # 1회 실행하면 그 다음부터는 불필요.
  print(f'{args.set} listing completed.\n{examples_path}')
  examples_list = dataset_util.read_examples_list(examples_path)
  
  for idx, example in enumerate(examples_list):
    if idx % 100 == 0:
      print(f'On image {idx} of {len(examples_list)}')
    path = os.path.join(annotations_dir, example + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
  
    tf_example = dict_to_tf_example(data, images_dir, label_map_dict,
                                    args.ignore_difficult_instances)
    writers[idx % args.num_shards].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()


if __name__ == '__main__':
  main()
