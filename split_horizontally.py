import cv2
import matplotlib.pyplot as plt
import numpy as np


def split_lincense_horizontally(image):
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_GB = cv2.GaussianBlur(gray, (3, 3), 0)
    # edges = cv2.Canny(gray_GB, 60, 120)   # img2tfrds can use the borders too
    ret, thr = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    character_num = 7
    for i in range(thr.shape[0]-1, -1, -1):
        is_delete = 0
        for j in range(character_num - 1, -1, -1):
            if sum(thr[i, j * thr.shape[1]//character_num:(j+1) *
                   thr.shape[1]//character_num]) == 0:
                is_delete += 1
        if is_delete > 1:
            thr = thr[:i]
        else:
            break
    thr = thr[::-1, :]
    for i in range(thr.shape[0]-1, -1, -1):
        is_delete = 0
        for j in range(character_num - 1, -1, -1):
            if sum(thr[i, j * thr.shape[1]//character_num:(j+1) *
                   thr.shape[1]//character_num]) == 0:
                is_delete += 1
        if is_delete > 1:
            thr = thr[:i]
        else:
            break
    thr = thr[::-1, :]
    if np.sum(thr[:1, :]) * 2 > 255 * thr.shape[1]:
        thr = 255 - thr
    # smooth bottom and top
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
    thr = thr[::-1]
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
    thr = thr[::-1]
    if thr[:, 0].all():
        cv2.floodFill(
            thr, np.zeros((thr.shape[0]+2, thr.shape[1]+2), dtype=np.uint8),
            (0, 0), 0)
    # floodFill to delete the vertical white space on both sides
    if thr[:, -1].all():
        cv2.floodFill(
            thr, np.zeros((thr.shape[0]+2, thr.shape[1]+2), dtype=np.uint8),
            (thr.shape[1]-1, 0), 0)
    plt.imshow(thr, cmap="gray")
    plt.title("floodFill")
    plt.show()

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr_without_circle = cv2.morphologyEx(
        cv2.dilate(thr, kernel), cv2.MORPH_CLOSE, kernel1)

    # cross line to kick out dot
    thr_without_circle[5, :] = 255

    _, cnts, _ = cv2.findContours(thr_without_circle,
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_premeter = [i for i in cnts if cv2.contourArea(i) < 100]
    cv2.fillPoly(thr_without_circle, small_premeter, 0)     # Denoising

    _, cnts, _ = cv2.findContours(thr_without_circle,
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_premeter = [i for i in cnts if cv2.contourArea(i) < 100]
    cv2.fillPoly(thr_without_circle, small_premeter, 255)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    thr_without_circle_and = cv2.bitwise_and(
        cv2.dilate(thr_without_circle, kernel2), thr)
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    # ax0.imshow(thr_without_circle, cmap="gray")
    # ax1.imshow(thr_without_circle_and, cmap="gray")
    # plt.show()

    # cut out the all-black borders
    for i in range(thr_without_circle_and.shape[1]):
        if sum(thr_without_circle_and[:, i]) > 0:
            thr_without_circle_and = thr_without_circle_and[:, i:]
            break
    for i in range(thr_without_circle_and.shape[1]-1, -1, -1):
        if sum(thr_without_circle_and[:, i]) > 0:
            thr_without_circle_and = thr_without_circle_and[:, :i+1]
            break

    # get split lines
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
    split_imgs = split_lincense_horizontally(image)
    print("split_imgs:", split_imgs)
    for i in split_imgs:
        plt.imshow(i, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
