r"""Convert raw COCO dataset to TFRecord for object_detection.
Please note that this tool creates sharded output files.
Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
import json
import os
import contextlib2
import tensorflow as tf
from utils import config as cfg


flags = tf.app.flags
tf.flags.DEFINE_string('train_image_dir', '/root/dataset/train2017/',
                       'Training image directory.')
tf.flags.DEFINE_string('val_image_dir', '/root/dataset/val2017/',
                       'Validation image directory.')
tf.flags.DEFINE_string('train_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/person_keypoints_train2017.json',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('val_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/root/dataset/tfrecord/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def create_tf_example(image,
                      annotations_list,
                      image_dir):
    """Converts image and annotations to a tf.Example proto.
    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_list:
        list of dicts with keys:
        ['keypoints', 'num_keypoints', 'id', 'image_id',
        'category_id', 'segmentation', 'area', 'bbox', 'iscrowd']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.

    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    keypoints = []
    bboxes = []
    result_dict = {'num_annotations_skipped': None, 'example': None, 'num_iscrowd': None}
    num_annotations_skipped = 0
    num_objects = 0
    num_iscrowd = 0
    for object_annotations in annotations_list:
        if object_annotations['iscrowd'] == 1:
            print("##")
            num_iscrowd += 1
            continue
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin = float(x)
        xmax = float(x + width)
        ymin = float(y)
        ymax = float(y + height)
        bboxes.extend([xmin, ymin, xmax, ymax])
        keypoints.extend(object_annotations['keypoints'])
        num_objects += 1

    if len(bboxes) != 0 and len(keypoints) != 0:

        non_obj = [0.0] * (cfg.COCO_MAX_PERSON_PER_PIC * 4 - len(bboxes))
        bboxes.extend(non_obj)
        non_kp = [0] * (cfg.COCO_MAX_PERSON_PER_PIC * cfg.COCO_NPOINTS * 3 - len(keypoints))
        keypoints.extend(non_kp)

        feature_example_list = []
        feature_dict = {
            'image/height':
                int64_feature(image_height),
            'image/width':
                int64_feature(image_width),
            'image/encoded':
                bytes_feature(encoded_jpg),
            'image/object/bboxes':
                float_list_feature(bboxes),
            'image/object/keypoints':
                int64_list_feature(keypoints)

        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        feature_example_list.append(example)
        result_dict = {'example': feature_example_list,
                       'num_annotations_skipped': num_annotations_skipped,
                       'num_iscrowd': num_iscrowd}
    return result_dict


def _create_tf_record_from_coco_annotations(
        annotations_file, image_dir, output_path, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.
    Args:
      annotations_file: JSON file containing bounding box annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: number of output file shards.
    """
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)  # json file
        images = groundtruth_data['images']

        annotations_index = {}
        # i = 0
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found {:<5} groundtruth annotations. Building annotations index.'
                .format(len(groundtruth_data['annotations'])))
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                num_keypoints = annotation['num_keypoints']

                if num_keypoints == 0:
                    continue
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
                # i+=1
                # print(annotation)
                # print(annotations_index.keys())
                # print(len(annotations_index[image_id]))
                # if i>10:
                #     break
            # image_id key

        # print('len_anno_index',len(annotations_index))
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        total_num_annotations_skipped = 0
        total_num_iscrowd = 0
        deal_img = []
        num = 0
        for idx, image in enumerate(images):

            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
            img_id = image['id']
            if img_id in deal_img:
                continue
            deal_img.append(img_id)
            annotations_list = annotations_index[img_id]
            result_dict = create_tf_example(image, annotations_list, image_dir)
            if not result_dict['example']:
                continue
            else:
                num += 1
                # if num == 5 or num == 4:
                #    print(num)
                tf_example, num_annotations_skipped, num_iscrowd \
                    = result_dict['example'], result_dict['num_annotations_skipped'], result_dict['num_iscrowd']
                total_num_annotations_skipped += num_annotations_skipped
                total_num_iscrowd += num_iscrowd
                shard_idx = idx % num_shards

                for i in range(len(tf_example)):
                    output_tfrecords[shard_idx].write(tf_example[i].SerializeToString())

        tf.logging.info('Finished writing, skipped %d annotations, skipped %d crowd annotations',
                        total_num_annotations_skipped, total_num_iscrowd)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_output_path = os.path.join(FLAGS.output_dir, 'train/', 'coco_train.record')

    if not tf.gfile.IsDirectory(os.path.join(FLAGS.output_dir, 'train/')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, 'train/'))

    val_output_path = os.path.join(FLAGS.output_dir, 'val/', 'coco_val.record')
    if not tf.gfile.IsDirectory(os.path.join(FLAGS.output_dir, 'val/')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, 'val/'))

    # _create_tf_record_from_coco_annotations(
    #     FLAGS.train_annotations_file,
    #     FLAGS.train_image_dir,
    #     train_output_path,
    #     num_shards=30)
    _create_tf_record_from_coco_annotations(
        FLAGS.val_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        num_shards=1)


if __name__ == '__main__':
    tf.app.run()
