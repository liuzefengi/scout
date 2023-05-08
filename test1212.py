
import cv2
import numpy as np



def LineFollower(cv_image):
    # def __init__(self,cv_image):


    # ROS Image's topic callback function


        cv2.imshow("cv_imcode_image", cv_image)

        height, width, channels = cv_image.shape
        descentre = 50
        rows_to_watch = 100
        # crop_img = cv_image
        # crop_img = cv_image[(height) / 4 + descentre:(height) / 4 + (descentre + rows_to_watch)][1:width]

        # convert from RGB to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)

        # yellow colour in HSV
        h_min, h_max, s_min, s_max, v_min, v_max = empty(0)
        lower_yellow = np.array([h_min, s_min, v_min])
        upper_yellow = np.array([h_max, s_max, v_max])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        cv2.imshow("MASK", mask)
###########
        # ret = method_3(mask)
        # cv2.imshow("ret", ret)
    ###########
        # Bitwise-and musk and original image
        res = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        # cv2.imshow("RES", res)

# 直接对输入图像转换为灰度图像，然后二值化
# def method_1(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return binary
#
# # 首先对输入图像进行降噪，去除噪声干扰，然后再二值化
# def method_2(image):
#     blurred = cv2.GaussianBlur(image, (3, 3), 0)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
#     t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return binary
#
# # 图像先均值迁移去噪声，然后二值化的图像
# def method_3(image):
#     blurred = cv2.pyrMeanShiftFiltering(image, 10, 100)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
#     t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     return binary

def empty(a):
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    return h_min, h_max, s_min, s_max, v_min, v_max


def main():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
    # line_follower_object = LineFollower()
    ctrl_c = False

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        retval, cv_image = cap.read()
        sss =  LineFollower(cv_image)

        if cv2.waitKey(5) >= 0:
          break







if __name__ == '__main__':
    main()