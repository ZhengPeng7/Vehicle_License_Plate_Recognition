import tensorflow as tf
import tfrecords2array
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict


def restore_lenet(char_classes):

    # ref
    recall_rate = OrderedDict().fromkeys([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z',
        '藏', '川', '鄂', '甘', '赣', '广', '桂', '贵', '黑',
        '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙',
        '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新',
        '渝', '豫', '粤', '云', '浙'
        ])
    for i in recall_rate.keys():
        recall_rate[i] = 0.0001
    class_count = recall_rate.copy()
    # y_train = []
    # x_train = []
    y_test = []
    x_test = []
    for char_class in char_classes:
        # train_data = tfrecords2array.tfrecord2array(
        #     r"./data_tfrecords/" + char_class + "_tfrecords/train.tfrecords")
        test_data = tfrecords2array.tfrecord2array(
            r"./data_tfrecords/" + char_class + "_tfrecords/test.tfrecords")
        # y_train.append(train_data[0])
        # x_train.append(train_data[1])
        y_test.append(test_data[0])
        x_test.append(test_data[1])
    for i in [y_test, x_test]:      # y_train, x_train,
        for j in i:
            print(j.shape)
    # y_train = np.vstack(y_train)
    # x_train = np.vstack(x_train)
    y_test = np.vstack(y_test)
    x_test = np.vstack(x_test)
    class_num = y_test.shape[-1]

    # print("x_train.shape=" + str(x_train.shape))
    print("x_test.shape=" + str(x_test.shape))
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
    correct_prediction = tf.equal(pred_class_index, tf.argmax(y_, 1))

    # 用平均值来统计测试准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
    batch_size_test = 64
    if not y_test.shape[0] % batch_size_test:
        epoch_test = y_test.shape[0] // batch_size_test
    else:
        epoch_test = y_test.shape[0] // batch_size_test + 1
    acc_test = 0
    class_sums = []
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
        print("{}-th pred_value={}".format(i, np.argmax(pred_value, 1)))
        # print("{}-th y_data_test={}".format(i, y_data_test))
        # print("\nCover:")
        # print("pred_value:", pred_value)
        # print("y_data_test:", y_data_test)
        # input()
        recall_sum = np.sum(cv2.bitwise_and(pred_value, y_data_test), axis=0)
        class_sum = np.sum(y_data_test, axis=0)
        class_sums.append(class_sum)
        # print(recall_sum)
        # input()
        for idx in range(len(recall_sum)):
            recall_rate[str(list(recall_rate.keys())[idx])] += recall_sum[idx]
            class_count[str(list(class_count.keys())[idx])] += class_sum[idx]
        # print(recall_rate)
        c = accuracy.eval(feed_dict={
            x: x_data_test, y_: y_data_test, keep_prob: 1.0})
        acc_test += c / epoch_test
    for i in list(recall_rate.keys()):
        recall_rate[i] /= class_count[i]

    print("recall_rate:\n", recall_rate)
    print("class_count:\n", class_count)
    print("class_sums:", np.sum(np.array(class_sums), axis=0))
    print("Restored acc_test={}".format(acc_test))
    return recall_rate


def main():
    # integers:         4679
    # alphabets:        9796
    # Chinese_letters:  3974
    # training_set : testing_set == 4 : 1
    test_lst = ['alphabets', 'integers', 'Chinese_letters']
    recall_rate = restore_lenet(test_lst)
    recall_rate_values = recall_rate.values()
    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(list(recall_rate_values), list(range(len(recall_rate_values))),
            '^')
    ax.hlines(list(range(len(recall_rate_values))), [0], recall_rate_values,
              lw=2)
    ax.set_xlabel('Recall rate')
    ax.set_ylabel('Idx of elem')
    ax.set_title('Statistics on Recall Rates')
    plt.show()


if __name__ == '__main__':
    main()
