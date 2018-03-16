import numpy as np
import tensorflow as tf
import time
import os
import cv2
from sklearn.utils import shuffle


# 图片存放位置
PATH_DES = [
    r'data_tfrecords/integers_tfrecords/',
    r'data_tfrecords/alphabets_tfrecords/',
    r'data_tfrecords/Chinese_letters_tfrecords/'
    ]
PATH_RES = [r'data/integers/',
            r'data/alphabets/',
            r'data/Chinese_letters/']

PATH = list(zip(PATH_RES, PATH_DES))
# transformation between integer <-> string
integers = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9
}
alphabets = {
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'I': 18,
    'J': 19,
    'K': 20,
    'L': 21,
    'M': 22,
    'N': 23,
    'O': 24,
    'P': 25,
    'Q': 26,
    'R': 27,
    'S': 28,
    'T': 29,
    'U': 30,
    'V': 31,
    'W': 32,
    'X': 33,
    'Y': 34,
    'Z': 35
}
provinces = {
    '藏': 36,
    '川': 37,
    '鄂': 38,
    '甘': 39,
    '赣': 40,
    '广': 41,
    '桂': 42,
    '贵': 43,
    '黑': 44,
    '沪': 45,
    '吉': 46,
    '冀': 47,
    '津': 48,
    '晋': 49,
    '京': 50,
    '辽': 51,
    '鲁': 52,
    '蒙': 53,
    '闽': 54,
    '宁': 55,
    '青': 56,
    '琼': 57,
    '陕': 58,
    '苏': 59,
    '皖': 60,
    '湘': 61,
    '新': 62,
    '渝': 63,
    '豫': 64,
    '粤': 65,
    '云': 66,
    '浙': 67
}
label_ref = [
    integers,
    alphabets,
    provinces
]


# 图片信息
IMG_HEIGHT = 28
IMG_WIDTH = 16
IMG_CHANNELS = 1
# NUM_TRAIN = 7000
NUM_VALIDARION = [sum([len(os.listdir(r + i))
                       for i in os.listdir(r)]) // 5 for r in PATH_RES]


# 读取图片
def read_images(path_res, label_ref, num_validation):
    imgs = []
    labels = []
    path_res_dirs = sorted(os.listdir(path_res))
    for i in path_res_dirs:
        paths_images = os.listdir(path_res + i)     # 本想排序的, 但是字符串排序效果不尽人意.
        t_lst = [''.join((path_res, i, '/', t)) for t in paths_images]
        paths_images = t_lst.copy()
        del t_lst
        for j in range(len(paths_images)):
            c = 0
            img = cv2.imread(paths_images[j], 0)
            img_blur = cv2.bilateralFilter(img, 3, 45, 45)
            img_current = cv2.resize(img_blur, (28, 28))
            ret, img_current_threshed = cv2.threshold(img_current,
                                                      127, 255,
                                                      cv2.THRESH_OTSU)
            h, w = img_current_threshed.shape
            t_c = np.array([[img_current_threshed[0][0],
                             img_current_threshed[0, w-1]],
                            [img_current_threshed[h-1, 0],
                             img_current_threshed[h-1, w-1]]])
            c = sum([(t_c[0, 0]//255), (t_c[1, 1]//255),
                     (t_c[0, 1]//255), (t_c[1, 0]//255)])
            if_reverse = sum([sum(img_current_threshed[0, :] // 255),
                              sum(img_current_threshed[:, w-1] // 255),
                              sum(img_current_threshed[h-1, :] // 255),
                              sum(img_current_threshed[:, 0] // 255)])\
                / ((h + w) * 2 + 4) > 0.5
            # if c >= 1:
            #     img_current_threshed = 255 - img_current_threshed
            if c > 2 or (c > 1 and if_reverse):
                img_current_threshed = 255 - img_current_threshed
            # img_current_threshed = img_current
            label_current = paths_images[j].split("/")[-2]
            # if i == '2':
            #     fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            #     ax0, ax1 = ax.ravel()
            #     ax0.imshow(img_current, cmap="gray")
            #     ax1.imshow(img_current_threshed, cmap="gray")
            #     plt.title(c)
            #     # print([img_current_threshed[0][0],
            #     #        img_current_threshed[0, w-1],
            #     #        img_current_threshed[h-1, 0],
            #     #        img_current_threshed[h-1, w-1]])
            #     plt.show()
            imgs.append((img_current_threshed // 255).astype(np.uint8))
            labels.append(np.uint8(label_ref[label_current]))
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0], -1)
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], -1)
    data = np.hstack((labels, imgs))
    data = shuffle(data)
    test_labels = data[:num_validation, 0]
    test_images = data[:num_validation, 1:]
    train_labels = data[num_validation:, 0]
    train_images = data[num_validation:, 1:]
    return train_labels, train_images, test_labels, test_images


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, filename):
    # 获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    print("num:", num)
    print("images.shape:", images.shape)
    # 输出TFRecord文件的文件名
    print('Writting', filename)
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        # 将图像矩阵转化为一个字符串
        img_raw = images[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
        # 将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')


def main():
    start_time = time.time()
    for i in range(len(PATH)):
        print('reading images from {} begin'.format(PATH_RES[i]))
        data = read_images(PATH_RES[i], label_ref[i], NUM_VALIDARION[i])
        train_labels, train_images, test_labels, test_images = data
        # Slice data here.
        print('convert to tfrecords into {} begin'.format(PATH_DES[i]))
        convert(train_images, train_labels, PATH_DES[i]+"train.tfrecords")
        convert(test_images, test_labels, PATH_DES[i]+"test.tfrecords")
    duration = time.time() - start_time
    print('Converting end , total cost = %d sec' % duration)


if __name__ == '__main__':
    main()
