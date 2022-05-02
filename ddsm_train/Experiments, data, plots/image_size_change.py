import cv2
import os
import pydicom
from PIL import Image

inputdir = '/home/adarshsehgal/Downloads/processed images/pos/'
outdir = '/home/adarshsehgal/Downloads/preprocessed images - 224x224/pos/'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list:
    image = Image.open(inputdir + f)
    new_image = image.resize((224, 224))
    new_image.save(outdir + f)