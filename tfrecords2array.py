import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import sys


def tfrecord2array(path_res):
    imgs = []
    lbls = []
    # print('tfrecords_files to be transformed:', path_res)
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer([path_res], num_epochs=1)

    # 从 TFRecord 读取内容并保存到 serialized_example 中
    _, serialized_example = reader.read(filename_queue)
    # 读取 serialized_example 的格式
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 解析从 serialized_example 读取到的内容
    labels = tf.cast(features['label'], tf.int64)
    images = tf.decode_raw(features['image_raw'], tf.uint8)

    # print('Extracting {} has just started.'.format(path_res))
    with tf.Session() as sess:
        # 启动多线程
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            try:
                label, img = sess.run([labels, images])
            except tf.errors.OutOfRangeError:
                print("Turn to next folder.")
                break
            img = (img > 0).astype(np.uint8).reshape(-1)
            imgs.append(img)
            lbls.append(label)
            clock_lines = ['-', '\\', '|', '/']

            sys.stdout.write(
                ''.join((str(np.array(lbls).shape[0]),
                         "-th sample in ",
                         path_res.split('/')[-2],
                         clock_lines[np.array(lbls).shape[0]//100 % 4],
                         '\r')))
            sys.stdout.flush()

        coord.request_stop()
        coord.join(threads)
    return to_categorical(np.array(lbls), num_classes=68), np.array(imgs)


def main():
    imgs, labels = tfrecord2array(
        r"./data_tfrecords/integers_tfrecords/test.tfrecords")
    print("imgs.shape:", imgs.shape)
    print("labels.shape:", labels.shape)


if __name__ == '__main__':
    main()
