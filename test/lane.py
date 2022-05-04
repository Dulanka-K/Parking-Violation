import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1- intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercepts(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(100,600),(900,800),(950,350),(500,350)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255) #create triangle
    masked_image = cv2.bitwise_and(image, mask) #& operation to get region of interest
    return masked_image

img = cv2.imread("../assets/pettah.jpg")
lane_image = np.copy(img)
canny_image = canny(lane_image)
cropped = roi(canny_image)
lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercepts(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

print(lines)
cv2.imshow("rsult", combo_image)
cv2.waitKey(0)
# plt.imshow(cropped)
# plt.show()

# cap = cv2.VideoCapture('../assets/pettah.mp4')
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped = roi(canny_image)
#     lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#     averaged_lines = average_slope_intercepts(frame, lines)
#     line_image = display_lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#
#     cv2.imshow("result", combo_image)
#     key = cv2.waitKey(1)
#     if key & 0xff == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
