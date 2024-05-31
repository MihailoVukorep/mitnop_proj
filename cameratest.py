#!/usr/bin/env python3

import cv2
import os
import sys
import pandas as pd
import numpy as np

from utils_main import *
from utils_selectmodel import selectmodel
model = selectmodel("all_v3_batch10000_epoch5.keras")

# video capture device
vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("can't open capture device")
    exit()

# windows
cv2.namedWindow("preview")

# make output dataframe
df = pd.DataFrame(reversed_class_mapping.items(), columns=["key", "char"])
clm = "prediction"
df[clm] = "0"

def image_filter(frame):
    img1 = cv2.resize(frame, (28, 28))
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    lower_px = 0
    upper_px = 110
    mask = cv2.inRange(img2, lower_px, upper_px)
    img3 = cv2.bitwise_and(img2, img2, mask=mask)
    img4 = np.expand_dims(img3, axis=-1)
    return img4


def detect(filtered_image):
    image_array = np.expand_dims(filtered_image, axis=0)
    predictions = model.predict(image_array, verbose=0)
    value = np.argmax(predictions, axis=1)
    value_id = value[0]
    label = reversed_class_mapping[value_id]
    values = predictions[0]
    df[clm] = [pred * 100 for pred in values]
    sorted_df = df.sort_values(by=clm, ascending=False)
    #print(str(sorted_df.to_string()))
    sys.stdout.write(f"\033[0;0H")
    sys.stdout.flush()
    print(sorted_df.head(n=6))

    
os.system("clear")

i = 0
while True:
    rval, frame = vc.read()

    if not rval:
        break

    procimg = image_filter(frame)

    frame[0:procimg.shape[0], 0:procimg.shape[1]] = procimg

    cv2.imshow("preview", frame)

    key = cv2.waitKey(1)

    if key == 27: # exit on ESC
        print("Closing webcam preview.")
        break
    elif key == 32: # detect on SPACE
        detect(procimg)
    
    i += 1
    if i > 20:
        i = 0
        detect(procimg)


vc.release()
cv2.destroyAllWindows()
