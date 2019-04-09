mport glob
from argparse import ArgumentParser

from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import ImageTk
from PIL import Image
import cv2
import numpy as np
import os
import shutil
import uuid
from sklearn.metrics.pairwise import pairwise_distances

from imutils import paths
import argparse
import cv2
import sys
 
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
 






if __name__ == "__main__":
    parser = ArgumentParser(description='Tool for performing homographies')

    parser.add_argument('--input', required=True,
        help='Input path for a catalog')
    parser.add_argument('--outputDir', required=True,
        help='Path for the output, will be wiped clean before any operation')
    args = vars(parser.parse_args())

    if not os.path.exists(args["outputDir"]):
        os.makedirs(args["outputDir"])
    cap = cv2.VideoCapture(args["input"])
   
    paths = glob.glob(os.path.join(args["input"],"*.jpg"))
    paths.append(glob.glob(os.path.join(args["input"],"*.png")))
    paths.append(glob.glob(os.path.join(args["input"],"*.jpeg")))
    paths.sort()
    liste = []

    for path in paths:
        image = cv2.imread(path)
        liste.append(variance_of_laplacian(image))
    

    for i in range(10):
        index = np.argmin(liste)
        bestImage = cv2.imread(paths[i])
        cv2.imwrite(os.path.join(args["output"],paths[i].split("/")[-1]))
        del liste[i]


    return True
    

       
    
    
