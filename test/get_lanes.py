import cv2
import numpy as np
import math

def get_lane_line_cordinates(img):
    # img = cv2.imread(r".\image.PNG", cv2.IMREAD_UNCHANGED)
    # img = cv2.imread("Image/lane_line_edge/frame_10.png", cv2.IMREAD_UNCHANGED)
    median = cv2.GaussianBlur(img, (3, 3), 0)
    gry = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    # low = np.array([24, 21, 202])
    # high = np.array([179, 255, 255])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # low = np.array([21, 23, 230])
    # high = np.array([39, 255, 255])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # low = np.array([0, 0, 113])
    # high = np.array([255, 26, 255])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # (hMin = 21, sMin = 23, vMin = 230), (hMax = 39, sMax = 255, vMax = 255)
    # (hMin = 154, sMin = 0, vMin = 0), (hMax = 179, sMax = 3, vMax = 227)
    # (hMin = 162, sMin = 0, vMin = 154), (hMax = 174, sMax = 107, vMax = 199)

    # low = np.array([0, 0, 151])
    # high = np.array([26, 38, 188])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # double line
    # low = np.array([0, 0, 162])
    # high = np.array([38, 128, 229])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # kadairi
    # low = np.array([0, 0, 216])
    # high = np.array([255, 17, 255])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # type 1
    # low = np.array([0, 0, 184])
    # high = np.array([255, 24, 255])
    # mask = cv2.inRange(gry, low, high)
    # edges = cv2.Canny(mask, 75, 150)

    # type 3
    low = np.array([0, 0, 230])
    high = np.array([255, 108, 255])
    mask = cv2.inRange(gry, low, high)
    edges = cv2.Canny(mask, 75, 150)

    return cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)



# img = cv2.imread("Image/lane_line_edge/frame_10.png", cv2.IMREAD_UNCHANGED)
# median = cv2.GaussianBlur(img, (5, 5), 1)
# gry = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
#
# low = np.array([0, 0, 151])
# high = np.array([26, 38, 188])
# mask = cv2.inRange(gry, low, high)
# edges = cv2.Canny(mask, 75, 150)
#
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=100)
#
# if lines is not None:
#     for line in lines:
#         # print(len(lines))
#         x1, y1, x2, y2 = line[0]
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
# cv2.imshow("result", img)
# cv2.waitKey(0)