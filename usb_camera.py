#!/usr/bin/env python3

# %% to run

import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)

    if key == 27: # exit on ESC
        print("Closing webcam preview.")
        break
    elif key == 32:
        file_name = f"image.png"
        cv2.imwrite(file_name, frame)
        print(f"Image {file_name} created.")
        print("Closing webcam preview.")
        break

vc.release()
cv2.destroyAllWindows()

# %%
