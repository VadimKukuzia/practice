# Python program to extract rectangular
# Shape using OpenCV in Python3
import cv2
import numpy as np

drawing = False  # true if mouse is pressed
ix, iy = -1, -1


# mouse callback function
def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            a = x
            b = y
            if a != x | b != y:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('Program to draw rectangular')
cv2.setMouseCallback('Program to draw rectangular', draw_shape)

while True:
    cv2.imshow('Program to draw rectangular', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
