# %% run
import cv2

# Open the first camera
cap1 = cv2.VideoCapture(0)  # Change 0 to the index of your first camera device

# Open the second camera
cap2 = cv2.VideoCapture(1)  # Change 1 to the index of your second camera device

# Check if the cameras opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Unable to open cameras")
    exit()

while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Check if frames were successfully read
    if not ret1 or not ret2:
        print("Error: Unable to read frames")
        break

    # Display frames from both cameras
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture objects and close all windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
# %%
