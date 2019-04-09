import glob
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
 


def findLessBlurry(images):
    bestImage = images[0]
    bestValue = -float("inf")
    for image in images :
        current = variance_of_laplacian(image)
        print("CURRENT",current)
        if bestValue < current:
            bestValue = current
            bestImage = image
    print("BEST",bestValue)
    return bestValue,image




if __name__ == "__main__":
    parser = ArgumentParser(description='Tool for performing homographies')

    parser.add_argument('--input', required=True,
        help='Input path for a catalog')
    parser.add_argument('--outputDir', required=True,
        help='Path for the output, will be wiped clean before any operation')
    parser.add_argument('--frequence', required=False, default = 10,
        help='Frequence wanted for each image')
    args = vars(parser.parse_args())

    if not os.path.exists(args["outputDir"]):
        os.makedirs(args["outputDir"])
    cap = cv2.VideoCapture(args["input"])
    compteur = 0
    ret, frame = cap.read()
    frameSize = int(args["frequence"])/2
    frameList = []
    frameList.append(frame)
    while(ret):
        
        # Capture frame-by-frame

        # if frame[0,0,0]==None:
            # break

        if frameSize == len(frameList): 
           
            bestValue,bestFrame = findLessBlurry(frameList)
            # Our operations on the frame come here
            # print(frame)
            # cv2.imshow('frame',frame)
            # Display the resulting frame
            cv2.imwrite(os.path.join(args["outputDir"],'frame{}.jpg'.format(compteur)),bestFrame)
            frameList = []
            frameSize = args["frequence"]
        compteur+=1
        if compteur>1000 :
            break

        ret, frame = cap.read()
        frameList.append(frame)
        
    if len(frameList)>0 :
        bestFrame = findLessBlurry(frameList)
        cv2.imwrite(os.path.join(args["outputDir"],'frame{}.jpg'.format(compteur)),bestFrame)
    
