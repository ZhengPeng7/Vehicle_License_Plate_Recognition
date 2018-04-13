import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform.radon_transform import radon


def detect_lincense(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_GB = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_GB, 60, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    edges_dilated = cv2.morphologyEx(cv2.dilate(edges, kernel),
                                     cv2.MORPH_CLOSE, kernel1)
    # gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    # gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # # subtract the y-gradient from the x-gradient
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # blurred = cv2.blur(gradient, (9, 9))
    # (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # closed = cv2.erode(closed, None, iterations=4)
    # edges_dilated = cv2.dilate(closed, None, iterations=4)
    plt.imshow(edges_dilated, cmap="gray")
    plt.show()
    img_cp = img.copy()
    (_, cnts, _) = cv2.findContours(edges_dilated.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cs = sorted(cnts, key=cv2.contourArea, reverse=True)
    # print("cs:", cs)
    shape_rate_criteria = 3.5
    min_shape_rate = 7
    img_area = img.shape[0] * img.shape[1]
    for c in cs:
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        # print("current box point:", box)
        sorted_box = np.array(sorted(box.tolist(), key=lambda x: x[0]))
        slope = ((((sorted_box[-1, 1] + sorted_box[-2, 1])/2) -
                  ((sorted_box[0, 1] + sorted_box[1, 1])/2)) /
                 (((sorted_box[-1, 0] + sorted_box[-2, 0])/2) -
                  ((sorted_box[0, 0] + sorted_box[1, 0])/2)))
        # print("sorted_box={}".format(sorted_box))
        # print("slope:", slope)
        height, width = (max(box[:, 1]) - min(box[:, 1]),
                         max(box[:, 0]) - min(box[:, 0]))
        shape_rate = width / height
        if (abs(shape_rate - shape_rate_criteria) <
           abs(min_shape_rate - shape_rate_criteria)) and 10 < height < 99 and\
           img_area / 300 < width*height < img_area / 20 and abs(slope) < 0.3:
            min_shape_rate = shape_rate
            extract_rect = box
            # print("shape_rate={}, distance={}".format(
            #     shape_rate, abs(shape_rate - shape_rate_criteria)))

        # draw a bounding box arounded the detected barcode and display the
        # image
        # cv2.drawContours(img_cp, [box], -1, (0, 255, 0), 3)
        # plt.imshow(img_cp)
        # plt.show()
    plate_area = img[min(extract_rect[:, 1]):max(extract_rect[:, 1]),
                     min(extract_rect[:, 0]):max(extract_rect[:, 0])]
    # plt.imshow(plate_area)
    # plt.show()

    return plate_area


def main():
    image = "./images/cars/car_2.jpg"
    plate_area = detect_lincense(image)
    plt.imshow(cv2.cvtColor(plate_area, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
