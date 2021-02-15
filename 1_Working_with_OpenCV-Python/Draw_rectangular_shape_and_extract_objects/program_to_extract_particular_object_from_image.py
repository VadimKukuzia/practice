# Write Python code here
# import the necessary packages
import cv2
import argparse
from datetime import datetime

# python program_to_extract_particular_object_from_image.py --image images/cats.jpg

# now let's initialize the list of reference point
ref_point = []
crop = False


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

    # construct the argument parser and parse the arguments


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)


def cut_n_save():
    global ref_point

    if len(ref_point) == 2:
        crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        cv2.imshow("crop_img", crop_img)
        time = datetime.strftime(datetime.now(), "%d.%m.%Y_%H.%M.%S")
        img = args['image'][args['image'].rfind('/') + 1:args['image'].rfind('.')]
        new_img_src = args['image'].replace(img, img + '_cropped_' + time)
        cv2.imwrite(new_img_src, crop_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # press 'r' to reset the window
    if key == ord("r"):
        image = clone.copy()

    if key == ord('q') or key == 27 or cv2.getWindowProperty("image", 0) < 0:
        break

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        cut_n_save()
        break

# close all open windows

