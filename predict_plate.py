import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import extract_figures
from collections import OrderedDict
import os
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import sys


myfont = FontProperties(fname='/usr/share/fonts/truetype/simhei.ttf', size=20)
rcParams['axes.unicode_minus'] = False


def predict_plate(plates):
    characters_ref = OrderedDict().fromkeys([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        '藏', '川', '鄂', '甘', '赣', '广', '桂', '贵', '黑',
        '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙',
        '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新',
        '渝', '豫', '粤', '云', '浙'
        ])
    characters_ref_keys = list(characters_ref.keys())
    # y_train = []
    # x_train = []
    y_test = np.zeros((7, 68), dtype=np.uint8)
    x_test = np.array(plates)
    # print("y_test.shape={}".format(y_test.shape))
    # print("x_test.shape={}".format(x_test.shape))

    class_num = y_test.shape[-1]
    sess = tf.InteractiveSession()

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, class_num])
    # 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层：卷积层
    conv1_weights = tf.get_variable(
        "conv1_weights",
        [5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
    conv1_biases = tf.get_variable("conv1_biases", [32],
                                   initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    # 移动步长为1, 使用全0填充
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))     # 激活函数Relu去线性化

    # 第二层：最大池化层
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # 第三层：卷积层
    conv2_weights = tf.get_variable(
        "conv2_weights",
        [5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64
    conv2_biases = tf.get_variable(
        "conv2_biases", [64], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1],
                         padding='SAME')
    # 移动步长为1, 使用全0填充
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层：最大池化层
    # 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # 第五层：全连接层
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    # 7*7*64=3136把前一层的输出变成特征向量
    fc1_biases = tf.get_variable(
        "fc1_biases", [1024], initializer=tf.constant_initializer(0.1))
    pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_biases)

    # 为了减少过拟合，加入Dropout层
    keep_prob = tf.placeholder(tf.float32)
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)

    # 第六层：全连接层
    fc2_weights = tf.get_variable("fc2_weights", [1024, class_num],
                                  initializer=tf.truncated_normal_initializer(
                                  stddev=0.1))
    # 神经元节点数1024, 分类节点10
    fc2_biases = tf.get_variable(
        "fc2_biases", [class_num], initializer=tf.constant_initializer(0.1))
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

    # 第七层：输出层
    # softmax
    y_conv = tf.nn.softmax(fc2)
    pred_class_index = tf.argmax(y_conv, 1)

    # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值
    # 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-class_num概率
    # correct_prediction = tf.equal(pred_class_index, tf.argmax(y_, 1))

    # 用平均值来统计测试准确率
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开始训练
    saver = tf.train.Saver()
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, './my_model/model.ckpt')
    # pred_value = sess.run([pred_class_index], feed_dict={
    #     x: x_test, y_: y_test, keep_prob: 1.0
    # })
    # print("pred_value=" + str(pred_value))
    # acc_test = sess.run(accuracy, feed_dict={
    #     x: x_test, y_: y_test, keep_prob: 1.0
    # })
    #
    batch_size_test = 1
    if not y_test.shape[0] % batch_size_test:
        epoch_test = y_test.shape[0] // batch_size_test
    else:
        epoch_test = y_test.shape[0] // batch_size_test + 1
    pred_values = []
    for i in range(epoch_test):
        if (i*batch_size_test % x_test.shape[0]) > (((i+1)*batch_size_test) %
                                                    x_test.shape[0]):
            x_data_test = np.vstack((
                x_test[i*batch_size_test % x_test.shape[0]:],
                x_test[:(i+1)*batch_size_test % x_test.shape[0]]))
            y_data_test = np.vstack((
                y_test[i*batch_size_test % y_test.shape[0]:],
                y_test[:(i+1)*batch_size_test % y_test.shape[0]]))
        else:
            x_data_test = x_test[
                i*batch_size_test % x_test.shape[0]:
                (i+1)*batch_size_test % x_test.shape[0]]
            y_data_test = y_test[
                i*batch_size_test % y_test.shape[0]:
                (i+1)*batch_size_test % y_test.shape[0]]
        # plt.imshow(x_data_test[0].reshape(28, 28), cmap="gray")
        # plt.show()
        # Calculate batch loss and accuracy
        pred_value = to_categorical(np.squeeze(
            sess.run([pred_class_index], feed_dict={
                   x: x_data_test, y_: y_data_test, keep_prob: 1.0})), 68)
        # print("{}-th pred_value={}".format(i, pred_value))
        pred_values.append(characters_ref_keys[(np.argmax(pred_value))])
    return pred_values


def main():

    # matplotlib 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    image = sys.argv[1]
    plate = extract_figures.extract_figures(image)
    pred_values = predict_plate(plate)
    pred_values.insert(2, '·')
    print("The License Plate is: {}".format(pred_values))
    plt.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    plt.title(u''.join(pred_values), fontproperties=myfont)
    plt.show()


if __name__ == '__main__':
    main()
