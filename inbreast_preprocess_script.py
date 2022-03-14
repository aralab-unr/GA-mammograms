import cv2
import os
import pydicom

inputdir = '/home/adarsh/Adarsh/INbreast Release 1.0/AllDICOMs/'
outdir = '/home/adarsh/Adarsh/INbreast Release 1.0/png images/'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list:   # remove "[:10]" to convert all images
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image