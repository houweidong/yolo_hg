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
tf.flags.DEFINE_string('keypoints_train_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/person_keypoints_train2017.json',
                       'KeyPoints Training annotations JSON file.')
tf.flags.DEFINE_string('keypoints_val_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json',
                       'KeyPoints Validation annotations JSON file.')

tf.flags.DEFINE_string('detection_train_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/instances_train2017.json',
                       'Detection Training annotations JSON file.')
tf.flags.DEFINE_string('detection_val_annotations_file',
                       '/root/dataset/annotations_trainval2017/annotations/instances_val2017.json',
                       'Detection Validation annotations JSON file.')


tf.flags.DEFINE_string('output_dir', '/root/dataset/tfrecord1/', 'Output data directory.')

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
                      annotations_dict,
                      image_dir):
    """Converts image and annotations to a tf.Example proto.
    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      annotations_dict:
        dicts of dicts with keys:
        {annotation_id:  'keypoints', 'num_keypoints', 'id', 'image_id',
        'category_id', 'segmentation', 'area', 'bbox', 'iscrowd'}
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
    num_keypoints = []
    bboxes = []
    result_dict = {'num_annotations_skipped': None, 'example': None, 'num_iscrowd': None}
    num_annotations_skipped = 0
    # num_objects = 0
    num_iscrowd = 0
    for object_annotations in annotations_dict.values():
        if object_annotations['iscrowd'] == 1:
            # print("##")
            num_iscrowd += 1
            num_annotations_skipped += 1
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
        if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
            print([xmin, ymin, xmax, ymax])
        bboxes.extend([xmin, ymin, xmax, ymax])
        keypoints.extend(object_annotations['keypoints'])
        num_keypoints.append(object_annotations['num_keypoints'])
        # num_objects += 1

    if len(bboxes) != 0 and len(keypoints) != 0:

        non_obj = [0.0] * (cfg.COCO_MAX_OBJECT_PER_PIC * 4 - len(bboxes))
        bboxes.extend(non_obj)

        non_kp = [0] * (cfg.COCO_MAX_OBJECT_PER_PIC * cfg.COCO_NPOINTS * 3 - len(keypoints))
        keypoints.extend(non_kp)

        non_num_kp = [0] * (cfg.COCO_MAX_OBJECT_PER_PIC - len(num_keypoints))
        num_keypoints.extend(non_num_kp)

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
                int64_list_feature(keypoints),
            'image/object/num_keypoints':
                int64_list_feature(num_keypoints),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        feature_example_list.append(example)
        result_dict = {'example': feature_example_list,
                       'num_annotations_skipped': num_annotations_skipped,
                       'num_iscrowd': num_iscrowd}
    return result_dict


def _create_tf_record_from_coco_annotations(
        det_annotations_file, kp_annotations_file, image_dir, output_path, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.
    Args:
      det_annotations_file: JSON file containing all categories bounding box annotations.
      kp_annotations_file: JSON file containing person bounding box and keypoints annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      num_shards: number of output file shards.
    """
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(det_annotations_file, 'r') as det_fid, \
            tf.gfile.GFile(kp_annotations_file, 'r') as kp_fid:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        # json file
        det_data = json.load(det_fid)
        kp_data = json.load(kp_fid)

        # det_data images should include kp_data images
        images = det_data['images']
        annotations_index = {}

        tf.logging.info(
            'Found {:<5} groundtruth annotations. Building annotations index.'
            .format(len(det_data['annotations'])))
        for annotation in det_data['annotations']:
            image_id = annotation['image_id']
            # num_keypoints = annotation['num_keypoints']
            if image_id not in annotations_index:
                annotations_index[image_id] = {}
            if annotation['category_id'] != 1:
                annotation['category_id'] = 0
            annotation['keypoints'] = [0] * 51
            annotation['num_keypoints'] = 0
            annotations_index[image_id][annotation['id']] = annotation

        for annotation_kp in kp_data['annotations']:
            image_id = annotation_kp['image_id']
            # if image_id not in annotations_index:
            #     print('there has images exists in kp_annotations but not exists in det_annotations')
            #     continue
            # if annotation_kp['id'] not in annotations_index[image_id]:
            #     print('#there has annotations exists in kp_annotations but not exists in det_annotations')
            #     continue
            annotations_index[image_id][annotation_kp['id']] = annotation_kp

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
            annotations_dict = annotations_index[img_id]
            if not annotations_dict:
                num += 1
                tf.logging.info('%d th missed images.',
                                num)
                continue
            result_dict = create_tf_example(image, annotations_dict, image_dir)
            if not result_dict['example']:
                print('not example')
                continue
            else:
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
    assert FLAGS.keypoints_train_annotations_file, '`keypoints_train_annotations_file` missing.'
    assert FLAGS.keypoints_val_annotations_file, '`keypoints_val_annotations_file` missing.'
    assert FLAGS.detection_train_annotations_file, '`detection_train_annotations_file` missing.'
    assert FLAGS.detection_val_annotations_file, '`detection_val_annotations_file` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_output_path = os.path.join(FLAGS.output_dir, 'train/', 'coco_train.record')

    if not tf.gfile.IsDirectory(os.path.join(FLAGS.output_dir, 'train/')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, 'train/'))

    val_output_path = os.path.join(FLAGS.output_dir, 'val/', 'coco_val.record')
    if not tf.gfile.IsDirectory(os.path.join(FLAGS.output_dir, 'val/')):
        tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, 'val/'))

    # _create_tf_record_from_coco_annotations(
    #     FLAGS.detection_train_annotations_file,
    #     FLAGS.keypoints_train_annotations_file,
    #     FLAGS.train_image_dir,
    #     train_output_path,
    #     num_shards=30)
    _create_tf_record_from_coco_annotations(
        FLAGS.detection_val_annotations_file,
        FLAGS.keypoints_val_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        num_shards=1)


if __name__ == '__main__':
    tf.app.run()
