#!/usr/bin/env python3

# %% run

import cv2
import numpy as np

def text_detection(image, padding=5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 0, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to find text regions
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
            #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Draw rectangles around text regions on the original image
    for (x, y, w, h) in text_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
        characters.append(thresh)

    return characters


# video capture device
vc = cv2.VideoCapture(0)
if not vc.isOpened():
    print("can't open capture device")
    exit()

# windows
cv2.namedWindow("preview")

while True:
    rval, frame = vc.read()

    if not rval:
        break

    cv2.imshow("preview", frame)

    detected_image, text_regions = text_detection(frame)
    cv2.imshow("preview2", detected_image)
    chars = character_segmentation(frame, text_regions)


    if len(chars) >= 1:

        # Resize characters to have the same height
        max_height = max(char.shape[0] for char in chars)
        resized_chars = [cv2.resize(char, (int(char.shape[1] * max_height / char.shape[0]), max_height)) for char in chars]

        combined_image = np.hstack(resized_chars)
        cv2.imshow("Combined Characters", combined_image)


    key = cv2.waitKey(1)

    if key == 27: # exit on ESC
        print("Closing webcam preview.")
        break

vc.release()
cv2.destroyAllWindows()

# %%
