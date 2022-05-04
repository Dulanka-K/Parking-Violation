import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("../assets/highway.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
# here we use background subtraction, less accurate
frame_cnt = 0

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame_cnt += 1
    # print('height:', height)
    # print('width', width)
    # Extract Region of interest
    # roi = frame[180: 640, 200: 550] #pettah
    roi = frame[340: 600, 500: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)  # bg subtracted image/frame
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # color value above threshold set to 255 (thresholding)
    # thresholding is a method of preprocessing

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:

        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    # 2. Object Tracking
    # print(detections)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) #add text
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 1) #draw rectangle

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()