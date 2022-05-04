from tracker import *
from polygon import *
import cv2
import numpy as np

draw = PolygonDrawer()
tracker = EuclideanDistTracker()

net = cv2.dnn.readNet("./weights/yolov3.weights","./weights/yolov3.cfg")
classes = []

with open("./weights/coco.names", "r") as f:
    classes = f.read().splitlines()

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture("./assets/borella.mp4")
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

success, img = cap.read()
pts = draw.getRegion(img)

mask = np.zeros(img.shape, np.uint8)
points = np.array(pts, np.int32)
points = points.reshape((-1, 1, 2))

# Draw polygon
mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
# mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # Used to find ROI
mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # Used for images displayed on the desktop



while True:
    timer = cv2.getTickCount()
    _, img1 = cap.read()
    # img = frame[180: 640, 200: 550]
    img = cv2.GaussianBlur(img1, (3,3),0)
    height, width, _ = img.shape #h=640 w=704

    # roi = img[180: 640, 200: 550]
    # print(roi)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    class_ids = []
    confidences = []
    boxes = []

    for output in layerOutputs:
        for detection in output:  # each detection have 85 parameters
            scores = detection[5:]  # first four elements are locations of bounding boxes,
                                    # next is the confidence, all other 80 are class probabs
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)  # detections are normalised, therefore we multiply
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # print(boxes)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #indexes of boxes that are not same object
    # print(indexes)
    detections = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            detections.append([x, y, w, h, i])

    # print(detections)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id, i = box_id
        box = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
        isInside = 0
        for k, l in box:
            inside = cv2.pointPolygonTest(points, (k, l), False)
            if inside != -1:
                isInside = 1

        if isInside:
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y-20), (x + (len(label)+len(str(id)))*15, y), color, -1)
            cv2.putText(img, label + " " + str(id), (x, y - 8), font, 1, (255, 255, 255), 1)

    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # cv2.putText(img, str(int(fps)), (650, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
    show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
    # out.write(show_image)
    cv2.imshow("Image", show_image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
