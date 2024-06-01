#!/usr/bin/env python3

# %% select model

import pandas as pd
import cv2
import numpy as np

# %% select model

from utils_main import *
from utils_selectmodel import selectmodel
model = selectmodel("all_v3_batch10000_epoch5.keras")
#model = selectmodel()

# make output dataframe
df = pd.DataFrame(reversed_class_mapping.items(), columns=["key", "char"])
clm = "prediction"
df[clm] = "0"

def text_detection(image, padding=3):
    imgcopy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    #cv2.imshow("blurred", blurred)
    edges = cv2.Canny(blurred, threshold1=60, threshold2=200, apertureSize=5, L2gradient=True)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
            text_regions.append((x, y, w, h))
            cv2.drawContours(imgcopy, [contour], -1, (0, 255, 0), 2)

    # draw rects
    for (x, y, w, h) in text_regions:
        cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (0, 0, 255), 2)


    cv2.imshow("rect", imgcopy)
    return image, text_regions


def character_segmentation(image, text_regions):
    characters = []
    for (x, y, w, h) in text_regions:
        # Extract each text region
        text_region = image[y:y+h, x:x+w]

        # Check if the text region is empty
        if text_region.size == 0:
            continue

        # Convert the text region to grayscale
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        img = np.expand_dims(thresh, axis=-1)

        characters.append(img)

    return characters


# video capture device
vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("can't open capture device")
    exit()

# windows
cv2.namedWindow("preview")

def display_chars_old(chars):
    chars.reverse()
    if len(chars) >= 1:
        # Resize characters to have the same height
        max_height = max(char.shape[0] for char in chars)
        resized_chars = [cv2.resize(char, (int(char.shape[1] * max_height / char.shape[0]), max_height)) for char in chars]

        combined_image = np.hstack(resized_chars)
        cv2.imshow("Combined Characters", combined_image)

def detect(chars):
    images = np.array(chars)
    print(images.shape, end=" -- ")
    predictions = model.predict(images, verbose=0)
    values = np.argmax(predictions, axis=1)
    text = [reversed_class_mapping[i] for i in values]
    print(text)


def resize_keep_channels(image, target_size=(28, 28)):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    resized_image = np.expand_dims(resized_image, axis=-1)
    return resized_image

def display_chars1(chars):
    chars.reverse()
    if len(chars) >= 1:
        # Resize characters to have the same height
        max_height = max(char.shape[0] for char in chars)
        resized_chars = [resize_keep_channels(char) for char in chars]

        combined_image = np.hstack(resized_chars)
        cv2.imshow("Combined Characters", combined_image)
        detect(resized_chars) 

i = 0
while True:
    rval, frame = vc.read()

    if not rval:
        break

    cv2.imshow("preview", frame)

    detected_image, text_regions = text_detection(frame)
    chars = character_segmentation(frame, text_regions)

    i += 1
    if i > 20:
        i = 0
        display_chars1(chars)

    key = cv2.waitKey(1)

    if key == 27: # exit on ESC
        print("Closing webcam preview.")
        break

vc.release()
cv2.destroyAllWindows()

# %%
