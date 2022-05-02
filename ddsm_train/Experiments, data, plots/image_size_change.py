import cv2
import os
import pydicom
from PIL import Image

inputdir = '/home/adarshsehgal/Downloads/processed images/neg/'
outdir = '/home/adarshsehgal/Downloads/preprocessed images - 1152x896/neg/'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list:
    image = Image.open(inputdir + f)
    new_image = image.resize((1152, 896))
    new_image.save(outdir + f)