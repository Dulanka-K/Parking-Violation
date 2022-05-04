import cv2
import os

import numpy as np

img = cv2.imread("../assets/pettah.jpg")
h, w, _ = img.shape
# print(int(h/2), int(w/2))
# For example,an edge detector output can be used as a mask to stop filling at edges

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (3,3),0)
blurcpy = cv2.GaussianBlur(img, (3,3),0)
# _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
# canny = cv2.Canny(blur, 50, 150)
print(img[600, 450])
cv2.floodFill(blur, None, (600, 450), (255, 0, 0), (4, 4, 4), (9, 9, 9))

# masked_image = cv2.bitwise_xor(blur, blurcpy)
# print(masked_image.flatten())
# mask = np.zeros_like(img)
# cv2.fillPoly(mask, masked_image, 255)

cv2.imshow("image", blur)
# cv2.imwrite("../assets/flood.jpg", blur)
cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', 'flood.jpg'), blur)

cv2.waitKey(0)
# import cv2
#
# image = cv2.imread("../assets/objects.png",0)
# height, width = image.shape
# cv2.imshow("image", image)
# cv2.waitKey(0)
# print(image.shape)
# nelem = 0
# for x in range(height):
#     for y in range(width):
#         if image[x,y] == 255:
#             nelem += 1
#             cv2.floodFill(image, None, (y,x), 150)
#
# print("Number of elements: ", nelem)
#
# cv2.imshow("image", image)
# cv2.waitKey(0)