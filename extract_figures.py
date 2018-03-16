import detect_lincense
import split_horizontally
import matplotlib.pyplot as plt
import cv2


def extract_figures(image):

    plate = detect_lincense.detect_lincense(image)
    # plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    # plt.show()
    split_figures = split_horizontally.split_lincense_horizontally(plate)
    for i in split_figures.copy():
        # plt.imshow(i, cmap="gray")
        # plt.show()
        gray_resized = cv2.resize(i, (28, 28), cv2.INTER_AREA)
        ret, thr = cv2.threshold(gray_resized, 127, 255, cv2.THRESH_OTSU)
        split_figures.pop(0)
        split_figures.append(thr.flatten())

    return split_figures


def main():
    image = "./images/cars/car_2.jpg"
    split_figures = extract_figures(image)
    for i in split_figures:
        plt.imshow(i.reshape(28, 28), cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
