import cv2
import matplotlib.pyplot as plt
import numpy as np
import detect_lincense


def split_lincense_horizontally(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    # 彩色图转灰度图, opencv读取图片通道顺序默认是bgr, 而非一半的rgb.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_GB = cv2.GaussianBlur(gray, (3, 3), 0)
    # edges = cv2.Canny(gray_GB, 60, 120)   # img2tfrds can use the borders too
    # 阈值化, params: 灰度图像, 将灰度值>90的像素升至255, 反之降至0, thr是阈值化后的二值图像.
    ret, thr = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    # 已知车牌共有7个字符+第三位置的"·"
    character_num = 7

    # 经验证, 这里原有的两个for循环并没有发挥功效, 原本的目的和下面一段相同, 都是为了上下两端的冗余黑背景的.
    # 之前代码不佳, 还请见谅.


    # 因为我在训练网络时, 尽可能地将字符调成了黑底白字,
    # 因此我这里设置: 最上面一行的白像素个数达到一半, 则认为该图片的背景是白的, 所以将其取反, 黑白互换.
    if np.sum(thr[:1, :]) * 2 > 255 * thr.shape[1]:
        thr = 255 - thr
    # smooth bottom and top
    # 这2个for循环的目的是将上下"磨平", 主要目的是去除上下两端冗余的黑底.
    # 未去除冗余的示意图为 "图0.png", 供参考比较.

    # 去冗余的整体思路: 从左至右遍历某行, 行内像素值跳变次数多, 则表示有有效字符, 不然就当做冗余. 剔除.
    for i in range(thr.shape[0]-1, -1, -1):
        jump_counter = 0
        prev_value = thr[i][0]  # 选取当前行的首个像素的值作为起始值.
        is_jump = 0     # the step must be larger than the criteria
        for j in thr[i]:
            is_jump += 1
            if j != prev_value:
                # 若行内的当前列的值j != 之前的像素值(即: 黑->白 或 白->黑)
                if is_jump > 3:     # if the step is satisfied to the condition
                    # 这里设一个is_jump的意思是, 为了防止像素值连续跳变(黑白黑白..., 这可能产生于边缘的凹凸处, 将图片放很大可看清, 见"图3.png")
                    jump_counter += 1    # 当前行跳变数 +1
                prev_value = j
                is_jump = 0
        if jump_counter < 12:
            thr = thr[:i]
        else:
            # 第一次出现不该删除的行, 则说明往里更内部的行皆为有效行.
            break
    thr = thr[::-1]
    # 刚刚是从下往上处理, 这里是先将图片上下颠倒后, 再从下往上处理, 再上下颠倒.
    for i in range(thr.shape[0]-1, -1, -1):
        jump_counter = 0
        prev_value = thr[i][0]
        is_jump = 0     # the step must be larger than the criteria
        for j in thr[i]:
            is_jump += 1
            if j != prev_value:
                if is_jump > 3:     # if the step is satisfied to the condition
                    jump_counter += 1
                prev_value = j
                is_jump = 0
        if jump_counter < 12:
            thr = thr[:i]
        else:
            break
    
    # floodFill to delete the vertical white space on both sides
    thr = thr[::-1]
    if thr[:, 0].all():
        # thr[:, 0].all() != 0 说明第一列全白, 这在汉字来说不可能, 因此只能是左右冗余的白段, 在(0, 0)和(shape[1]-1, 0)两点浸水填充一下, 消去. 
        cv2.floodFill(
            thr, np.zeros((thr.shape[0]+2, thr.shape[1]+2), dtype=np.uint8),
            (0, 0), 0)
    if thr[:, -1].all():
        cv2.floodFill(
            thr, np.zeros((thr.shape[0]+2, thr.shape[1]+2), dtype=np.uint8),
            (thr.shape[1]-1, 0), 0)
    plt.imshow(thr, cmap="gray")
    plt.title("floodFill")
    plt.show()

    # Morphology
    # 使用形态学变换, 是字符膨胀"成团"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr_without_circle = cv2.morphologyEx(
        cv2.dilate(thr, kernel), cv2.MORPH_CLOSE, kernel1)

    # cross line to kick out dot
    # 通过划线来剔除车牌中的"·"
    thr_without_circle[5, :] = 255
    # 在第5行划白线, 由于第5行很高, 因此不会接触到圆点, 从而可以把正常字符都连接起来, 形成一个大连通域.
    # 然后消去小的连通域 -- 正常字符的主体都是那个大连通域的一部分, 因此不受影响, 而"·"未能连接, 因此属于小面积轮廓, 将被消去(将它填充背景色).

    _, cnts, _ = cv2.findContours(thr_without_circle,
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_area = [i for i in cnts if cv2.contourArea(i) < 100]
    cv2.fillPoly(thr_without_circle, small_area, 0)     # Denoising
    # 通过上面三行, 我们获取了用于消去"·"的掩码--图4.

    # 然后让图4与图3做 与操作, 即可得到缺失"·"的原图.
    # 原来这里有多余的三行, 无效, 故删之.
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thr_without_circle_and = cv2.bitwise_and(
        cv2.dilate(thr_without_circle, kernel2), thr)
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    # ax0.imshow(thr_without_circle, cmap="gray")
    # ax1.imshow(thr_without_circle_and, cmap="gray")
    # plt.show()

    # cut out the all-black borders
    # 通过定位每个字符的最小纵坐标和最大纵坐标, 去除字符之间的黑段, 将其分别提取出来.
    for i in range(thr_without_circle_and.shape[1]):
        if sum(thr_without_circle_and[:, i]) > 0:
            thr_without_circle_and = thr_without_circle_and[:, i:]
            break
    for i in range(thr_without_circle_and.shape[1]-1, -1, -1):
        if sum(thr_without_circle_and[:, i]) > 0:
            thr_without_circle_and = thr_without_circle_and[:, :i+1]
            break

    # get split lines
    # 类似于之前的去除上下端冗余黑段, 这里也是从左往右扫获取所有字符的最小横坐标,从右往左扫获取all x_min
    # 2者zip一下便获得了((x0_min, x0_max), (x1_min, x1_max), ..., (x6_min, x6_max)), 据此截段, 作为结果.
    vertical_lines_0 = []
    vertical_lines_1 = []
    is_append = 1
    for i in range(thr_without_circle_and.shape[1]):
        if sum(thr_without_circle_and[:, i]) == 0:
            if is_append:
                vertical_lines_0.append(i)
                is_append = 0
        else:
            is_append = 1
    is_append = 1
    for i in range(thr_without_circle_and.shape[1] - 1, -1, -1):
        if sum(thr_without_circle_and[:, i]) == 0:
            if is_append:
                vertical_lines_1.append(i)
                is_append = 0
        else:
            is_append = 1

    # split out the individual character into split_imgs
    black_spaces = list(zip(vertical_lines_0, sorted(vertical_lines_1)))
    white_spaces = black_spaces.copy()
    white_spaces = np.array(black_spaces).flatten().tolist()
    white_spaces.append(thr_without_circle_and.shape[1])
    white_spaces.insert(0, 0)
    # print("white_spaces:", white_spaces)
    split_imgs = []
    for i in range(0, len(white_spaces), 2):
        split_imgs.append(
            thr_without_circle_and[:, white_spaces[i]:white_spaces[i+1]])

    return split_imgs


def main():
    image = "./images/cars/car_0.jpg"
    plate = detect_lincense.detect_lincense(image)
    split_imgs = split_lincense_horizontally(plate)
    # print("split_imgs:", split_imgs)
    for i in split_imgs:
        plt.imshow(i, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
