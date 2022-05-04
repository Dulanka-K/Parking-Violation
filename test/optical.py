import cv2
from tracker2 import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("../assets/borella.mp4")
back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=400, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
frameRate = cap.get(cv2.CAP_PROP_FPS)
print(frameRate)

while True:
    _, frame = cap.read()
    # frame = cv2.resize(frame, (600, 600))
    fgmask = back_sub.apply(frame)
    # bg = back_sub.getBackgroundImage()


    # cv2.imshow("FG mask", fgmask)
    # cv2.imshow("Background", bg)

    # try changing dilation and erosion (uda yata maru karanna)
    eroded = cv2.erode(fgmask, kernel)
    dilated = cv2.dilate(eroded, kernel)
    cv2.imshow("Blob mask", dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if (area > 1000) & (area < 2000):
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) #add text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) #draw rectangle

    cv2.imshow("Frame", frame)
    #tracking

    key = cv2.waitKey(50)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()