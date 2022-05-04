import cv2
import os
import numpy as np
import time
from tracker import *
from direction import *

tracker = EuclideanDistTracker()
dirIdentifier = DirectionIdentifier()

# Load Yolo
net = cv2.dnn.readNet("../weights/yolov3.weights", "../weights/yolov3.cfg")
classes = []
with open("../weights/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

reqClasses = [1, 2, 3, 5, 7, 9]
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture("../assets/pettah.mp4")
clrlight_frame = 50
frame_cnt = 0
clrlight_boxes = []
clrlight_confidences = []
fin_clrlight_boxes = []

while True:
    start = time.time()
    _, img = cap.read()
    frame_cnt += 1
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id in reqClasses:
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    # print(boxes)

    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, w, h, i])
            if frame_cnt < clrlight_frame:
                if class_ids[i] == 9: #color light
                    clrlight_boxes.append([x, y, w, h])
                    clrlight_confidences.append(confidences[i])

    if clrlight_boxes:
        if frame_cnt == clrlight_frame:
            clr_index = cv2.dnn.NMSBoxes(clrlight_boxes, clrlight_confidences, 0.5, 0.4)
            fin_clrlight_boxes = clr_index.flatten()
        elif frame_cnt > clrlight_frame:
            for i in fin_clrlight_boxes:
                x, y, w, h = clrlight_boxes[i]
                # print(clrlight_boxes[i])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    boxes_ids, centers = tracker.update(detections)
    # print(centers)
    for box_id in boxes_ids:
        x, y, w, h, id, i = box_id
        # box = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

        label = str(classes[class_ids[i]])
        # confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # print(centers[id])
        direction = ''
        if len(centers[id]) >= tracker.trajectory_len:
            direction = dirIdentifier.getDirections(centers[id])
            cv2.circle(img, centers[id][-1], 2, (0, 0, 255), -1)
            cv2.polylines(img, [np.int32(centers[id])], False, (0, 255, 0))
        cv2.rectangle(img, (x, y - 20), (x + (len(label) + len(str(id))) * 15, y), color, -1)
        cv2.putText(img, label + " " + str(id) + " " + direction, (x, y - 8), font, 1, (255, 255, 255), 2)

    cv2.putText(img, "Frame: " + str(frame_cnt), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(img, f"{fps:.2f} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
