import cv2
import imutils
import numpy as np
import joblib

pts = []  # For storage points


# 统一的：mouse callback function
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to select point
        pts.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:  #Right click to cancel the last selected point
        pts.pop()

    if event == cv2.EVENT_MBUTTONDOWN:  # Middle button to draw outline
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        # Draw polygon
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # Used to find ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # Used for images displayed on the desktop

        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # Draw the last point in pts
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # Draw a line
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y Is the coordinates of the mouse click
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)


# Create image and window and bind window to callback function
img = cv2.imread("../assets/pettah.png")
img = imutils.resize(img, width=500)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
print("[INFO] Click the left button: select the point, click the right button: delete the last selected point, click the middle button: confirm the ROI area")
print("[INFO] Press ‘S’ to confirm the selected area and save")
print("[INFO] Press ESC to exit")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {
            "ROI": pts
        }
        joblib.dump(value=saved_data, filename="config.pkl")
        print("[INFO] ROI coordinates have been saved locally.")
        break
cv2.destroyAllWindows()
