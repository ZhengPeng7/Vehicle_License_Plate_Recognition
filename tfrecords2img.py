import os
import shutil
import tensorflow as tf
import time
import sys
import cv2


# 图片存放位置
PATH_RES = [
    r'data_tfrecords/integers_tfrecords/train.tfrecords',
    r'data_tfrecords/integers_tfrecords/test.tfrecords',
    r'data_tfrecords/alphabets_tfrecords/train.tfrecords',
    r'data_tfrecords/alphabets_tfrecords/test.tfrecords',
    r'data_tfrecords/Chinese_letters_tfrecords/train.tfrecords',
    r'data_tfrecords/Chinese_letters_tfrecords/test.tfrecords'
    ]
PATH_DES = [
    r'imgs_from_tfrecords/integers/train/',
    r'imgs_from_tfrecords/integers/test/',
    r'imgs_from_tfrecords/alphabets/train/',
    r'imgs_from_tfrecords/alphabets/test/',
    r'imgs_from_tfrecords/Chinese_letters/train/',
    r'imgs_from_tfrecords/Chinese_letters/test/'
    ]

PATH = list(zip(PATH_RES, PATH_DES))


def tfrecord2jpg(path_res, path_des):
    print('tfrecords_files to be transformed:', path_res)
    reader = tf.TFRecordReader()
    start_time = int(time.time())
    prev_time = start_time
    idx = 0

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
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.cast(features['label'], tf.int64)

    print('Extracting {} has just started.'.format(path_res))
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
            cv2.imwrite(path_des+"_"+str(idx)+"_"+str(label)+'.jpg', img)
            idx += 1
            current_time = int(time.time())
            lasting_time = current_time - start_time
            interval_time = current_time - prev_time
            if interval_time >= 0.1:
                sys.stdout.flush()
                sys.stdout.write("\rGenerating the {}-th image: {},\
                                    lasting {} seconds".format(
                                    idx,
                                    path_des +
                                    str(idx) + '_' +
                                    str(label) + '.jpg',
                                    lasting_time))
                prev_time = current_time
        coord.request_stop()
        coord.join(threads)


def main():
    # get empty directory
    for i in range(len(PATH)):
        if os.path.isdir(PATH_DES[i]):
            if os.listdir(PATH_DES[i]):
                shutil.rmtree(PATH_DES[i])
                os.mkdir(PATH_DES[i])
        else:
            print(PATH_DES[i])
            os.mkdir(PATH_DES[i])
        tfrecord2jpg(PATH_RES[i], PATH_DES[i])


if __name__ == "__main__":
    main()
