from get_lanes import *
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import math

# image_read = cv2.imread('E:/Python/Image/lane_line_edge/line_type_1.jpg', cv2.IMREAD_UNCHANGED)
image_read = cv2.imread('../assets/pettah.jpg', cv2.IMREAD_UNCHANGED)
lines = get_lane_line_cordinates(image_read)
# directory = r'E:\Python\Image\lane_line_edge'
video = cv2.VideoCapture("../assets/pettah.mp4")
# os.chdir(directory)
originalMask = [];

# lines = getLaneLineCordinates()

def make_line_cleaner(h, w, img):
    for x in range(h):
        for y in range(w):
            if img[x][y] > 0:
                img[x][y] = 255

    return img

def getNumOfPeakPixels(h, w, newMask):
    x_axix = []
    y_axix = []

    for x in range(w):
        count = 0
        for y in range(h-5, h):
            valu = 0;
            if newMask[y][x] == 255:
                valu = newMask[y][x]
            count = count + valu
            # count = count + newMask[y][x]
        x_axix.append(x + 1)
        y_axix.append(int(count / 5))

    # prev = y_axix[0] or 0.001
    # threshold = 0.5
    # peaks = []
    #
    # for num, i in enumerate(y_axix[1:], 1):
    #     if (i - prev) / prev > threshold:
    #         peaks.append(num)
    #     prev = i or 0.001
    x = np.array(x_axix)
    y = np.array(y_axix)
    peaks, _ = find_peaks(y, height=20, distance=10)
    # print(y_axix)
    # plt.plot(x_axix, y_axix)
    # plt.plot(peaks, y[peaks], "x")
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title("A simple line graph")
    # plt.show()
    # print(len(peaks))
    # print(y_axix)
    return math.pow(len(peaks), 2)

# mouse event handle
circles = []
def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(circles) < 4:
            print((x, y))
            circles.append((x, y))

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

while True:
    # read frams from the video
    ret, frame = video.read()
    frame = image_read

    if not ret:
        video = cv2.VideoCapture("../assets/pettah.mp4")
        continue

    if lines is not None:
        for line in lines:
            # print(len(lines))
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    h, w = frame.shape[:2]
    mask = np.zeros((h,w, 3), dtype="uint8")

    if lines is not None:
        for line in lines:
            # print(len(lines))
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 5)


    cv2.circle(frame, (441, 223), 1, (255, 0, 0), 1)

    for point in circles:
        cv2.circle(frame, point, 5, (0,0,255), -1)

    if len(circles) == 4:
        for x in range(4):
            if x < 3:
                cv2.line(frame, circles[x], circles[x+1], (0, 0, 255))
            elif x == 3:
                cv2.line(frame, circles[x], circles[0], (0, 0, 255))

    if len(circles) == 4:

        gry = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

        low = np.array([0, 103, 255])
        high = np.array([179, 255, 255])
        mask = cv2.inRange(gry, low, high)
        edges = cv2.Canny(mask, 75, 150)

        # print([[circles[0][0], circles[0][1]], [circles[1][0], circles[1][1]], [circles[3][0], circles[3][1]], [circles[2][0], circles[2][1]] ])
        pts1 = np.float32([[circles[0][0], circles[0][1]], [circles[1][0], circles[1][1]], [circles[3][0], circles[3][1]], [circles[2][0], circles[2][1]] ])
        pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])

        mtrix = cv2.getPerspectiveTransform(pts1, pts2)

        result = cv2.warpPerspective(mask, mtrix, (400, 600))
        h, w = result.shape[:2]
        peak_pixel = 0
        n = int(h / 5)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        clean_line = make_line_cleaner(h, w, closing)
        bure = cv2.medianBlur(closing, 9)
        for x in range(n):
            peak_pixel = peak_pixel + getNumOfPeakPixels((x + 1) * 5, w, bure)
        print(peak_pixel / n)
        cv2.imshow("result1", mtrix)
        cv2.imshow("road", bure)

    cv2.imshow("Frame", frame)
    # cv2.imshow("result1", result)
    cv2.imshow("mask", mask)
    cv2.imshow("read", image_read)
    key = cv2.waitKey(25)

    if key == 27:
        break
video.release()
cv2.destroyAllWindows()