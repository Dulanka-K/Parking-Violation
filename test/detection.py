import cv2
import os
import numpy as np
import time
from tracker import *
from direction import *
from get_lanes import *
from hough_bundler import *

tracker = EuclideanDistTracker()
dirIdentifier = DirectionIdentifier()
bundler = HoughBundler(min_distance=10, min_angle=5)

# Load Yolo
net = cv2.dnn.readNet("../weights/yolov3.weights", "../weights/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("../weights/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

reqClasses = [1, 2, 3, 5, 7, 9]
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture("../assets/pettah.mp4")
clrlight_frame = 50
dir_detect_frame = 130
frame_cnt = 0
clrlight_boxes = []
clrlight_confidences = []
fin_clrlight_boxes = []
min_trajectory_len = 50
traj_tail_len = 10

_, init_image = cap.read()
init_height, init_width, _ = init_image.shape
upmask = np.zeros((init_height, init_width), np.uint8)
downmask = np.zeros((init_height, init_width), np.uint8)
upcenters = []
downcenters = []

upcontour = np.zeros((init_height, init_width, 3), np.uint8)
downcontour = np.zeros((init_height, init_width, 3), np.uint8)
upROI = np.zeros((init_height, init_width, 3), np.uint8)
downROI = np.zeros((init_height, init_width, 3), np.uint8)
dlines = []
ulines = []

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

updetected = False
downdetected = False

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)

cap.set(cv2.CAP_PROP_POS_FRAMES, 530)

def closestPoint(p1, p2, p3):
    [x1, y1] = p1
    [x2, y2] = p2
    [x3, y3] = p3
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return [x1+a*dx, y1+a*dy]

while True:
    start = time.time()
    _, img = cap.read()
    frame_cnt += 1
    height, width, channels = img.shape

    # Loading image
    # img = cv2.imread("../assets/dog.jpg")
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)

    if frame_cnt < dir_detect_frame:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        # print(mask)
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # for b in blob:
        #     for n, img_blob in enumerate(b):
        #         cv2.imshow(str(n), img_blob)

        net.setInput(blob)
        outs = net.forward(output_layers)
        # print(outs)

        # Showing informations on the screen
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
                del reqClasses[5]
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
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            # print(centers[id])
            direction = ''
            mask = np.zeros((height + 2, width + 2), np.uint8)

            if len(centers[id]) >= tracker.trajectory_len:
                dist = math.hypot(centers[id][-1][0] - centers[id][0][0], centers[id][-1][1] - centers[id][0][1])
                if dist > min_trajectory_len:
                    direction = dirIdentifier.getDirections(centers[id])
                    # print(centers[id])
                    cv2.circle(img, centers[id][-1], 2, (0, 0, 255), -1)
                    cv2.polylines(img, [np.int32(centers[id])], False, (0, 255, 0), 1)
                    for i, center in enumerate(centers[id]):
                        if i < traj_tail_len:
                            cv2.floodFill(blur, mask, center, (255, 0, 0), (4, 4, 4), (9, 9, 9), floodflags)
                    mask = mask[1:height + 1, 1: width + 1]
                    if direction == 'up':
                        upmask = cv2.add(upmask, mask)
                        upcenters.append(centers[id])
                    elif direction == 'down':
                        downmask = cv2.add(downmask, mask)
                        downcenters.append(centers[id])

            # cv2.rectangle(img, (x, y - 20), (x + (len(label) + len(str(id))) * 15, y), color, -1)
            # cv2.putText(img, label + " " + str(id) + " " + direction, (x, y - 8), font, 1, (255, 255, 255), 2)

    if frame_cnt == dir_detect_frame:
        img_copy = img.copy()
        # updilated = cv2.dilate(upmask, kernel)
        # downdilated = cv2.dilate(downmask, kernel)

        upcontours, uphierarchy = cv2.findContours(upmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        downcontours, downhierarchy = cv2.findContours(downmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        sorted_up = sorted(upcontours, key=cv2.contourArea, reverse=True)
        if sorted_up:
            # upcontour = cv2.polylines(upcontour, [sorted_up[0]], True, (255, 255, 255), 2)
            updetected = True
            uphull = cv2.convexHull(sorted_up[0])
            cv2.drawContours(upcontour, [uphull], -1, (255, 255, 255), 4)
            upcontour = cv2.fillPoly(upcontour, [uphull], (255, 255, 255))
            upROI = cv2.bitwise_and(upcontour, img_copy)
            uplines = get_lane_line_cordinates(upROI)
            if uplines is not None:
                ulines = bundler.process_lines(uplines)
                print(ulines)

        sorted_down = sorted(downcontours, key=cv2.contourArea, reverse=True)
        if sorted_down:
            # downcontour = cv2.polylines(downcontour, [sorted_down[0]], True, (255, 255, 255), 2)
            downdetected = True
            downhull = cv2.convexHull(sorted_down[0])
            cv2.drawContours(downcontour, [downhull], -1, (255, 255, 255), 4)
            downcontour = cv2.fillPoly(downcontour, [downhull], (255, 255, 255))
            downROI = cv2.bitwise_and(downcontour, img_copy)
            downlines = get_lane_line_cordinates(downROI)
            if downlines is not None:
                dlines = bundler.process_lines(downlines)
                print(dlines)

        nearest_up = {} #color lights nearest to up and down directions
        nearest_down = {}
        for i in fin_clrlight_boxes:
            x, y, w, h = clrlight_boxes[i]
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            if sorted_up:
                dist_frm_up = cv2.pointPolygonTest(sorted_up[0], (cx, cy), True)
                dist_frm_up = int(abs(dist_frm_up))
                if bool(nearest_up):
                    if list(nearest_up.values())[0] > dist_frm_up:
                        nearest_up = {}
                        nearest_up[i] = dist_frm_up
                else:
                    nearest_up[i] = dist_frm_up

            if sorted_down:
                dist_frm_down = cv2.pointPolygonTest(sorted_down[0], (cx, cy), True)
                dist_frm_down = int(abs(dist_frm_down))
                if bool(nearest_down):
                    if list(nearest_down.values())[0] > dist_frm_down:
                        nearest_down = {}
                        nearest_down[i] = dist_frm_down
                else:
                    nearest_down[i] = dist_frm_down

    elif frame_cnt > dir_detect_frame:
        # img_copy = img.copy()
        if updetected:
            img = cv2.addWeighted(src1=img, alpha=1, src2=upcontour, beta=0.3, gamma=0)
            up_id = next(iter(nearest_up))
            x, y, w, h = clrlight_boxes[up_id]
            cv2.putText(img, "up_lt", (x, y - 8), font, 1, (255, 255, 255), 2)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            cv2.circle(img, [int(cx), int(cy)], 2, (0, 255, 255), -1)
            for centerpoints in upcenters:
                [x, y] = closestPoint(centerpoints[0], centerpoints[-1], [cx, cy])
                cv2.circle(img, [int(x), int(y)], 2, (0, 0, 255), -1)
                cv2.line(img, centerpoints[0], centerpoints[-1], (255, 0, 0), 2)
            # add line detection here
            if ulines is not None:
                for line in ulines:
                    # print(len(lines))
                    x1, y1, x2, y2 = line[0]
                    cv2.line(upROI, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(upROI, [x1, y1], 2, (0, 255, 255), -1)
                    cv2.circle(upROI, [x2, y2], 2, (0, 255, 255), -1)

        if downdetected:
            img = cv2.addWeighted(src1=img, alpha=1, src2=downcontour, beta=0.3, gamma=0)
            down_id = next(iter(nearest_down))
            x, y, w, h = clrlight_boxes[down_id]
            cv2.putText(img, "down_lt", (x, y - 8), font, 1, (255, 255, 255), 2)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            cv2.circle(img, [int(cx), int(cy)], 2, (0, 255, 255), -1)
            for centerpoints in downcenters:
                [x, y] = closestPoint(centerpoints[-1], centerpoints[0], [cx, cy])
                cv2.circle(img, [int(x), int(y)], 2, (0, 0, 255), -1)
                cv2.line(img, centerpoints[0], centerpoints[-1], (255, 0, 0), 2)
            # add line detection here
            if dlines is not None:
                for line in dlines:
                    # print(len(lines))
                    x1, y1, x2, y2 = line[0]
                    cv2.line(downROI, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(downROI, [x1, y1], 2, (0, 255, 255), -1)
                    cv2.circle(downROI, [x2, y2], 2, (0, 255, 255), -1)

        cv2.imshow("Up ROI", upROI)
        cv2.imshow("Down ROI", downROI)

    cv2.putText(img, "Frame: " + str(frame_cnt), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(img, f"{fps:.2f} FPS", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Up Mask", upmask)
    # cv2.imshow("Down Mask", downmask)
    # cv2.imshow("Up Contour", upcontour)
    # cv2.imshow("Down Contour", downcontour)


    key = cv2.waitKey(150)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# for i in range(len(boxes)):
#     if i in indexes:
#         x, y, w, h = boxes[i]
#         label = str(classes[class_ids[i]])
#         color = colors[class_ids[i]]
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#         cv2.rectangle(img, (x, y - 20), (x + len(label) * 15, y), color, -1)
#         cv2.putText(img, label, (x, y - 8), font, 1, (255, 255, 255), 1)


# if updetected:
#     up_id = next(iter(nearest_up))
#     x, y, w, h = clrlight_boxes[up_id]
#     cx = (x + x + w) // 2
#     cy = (y + y + h) // 2
#     for centerpoints in upcenters:
#         closestPt = closestPoint(centerpoints[0], centerpoints[-1], [cx, cy])
#         cv2.circle(img, closestPt, 2, (0, 0, 255), -1)
#
# if downdetected:
#     down_id = next(iter(nearest_down))
#     x, y, w, h = clrlight_boxes[down_id]
#     cx = (x + x + w) // 2
#     cy = (y + y + h) // 2
#     for centerpoints in downcenters:
#         closestPt = closestPoint(centerpoints[0], centerpoints[-1], [cx, cy])
#         cv2.circle(img, closestPt, 2, (0, 0, 255), -1)