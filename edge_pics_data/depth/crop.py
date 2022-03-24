import cv2
import os

for filename in os.listdir('./'):
  if filename.endswith(".png"):
        img = cv2.imread(filename)
        crop_img = img[33:252, 72:366]
        cv2.imwrite("crop/cropped"+filename, crop_img)
