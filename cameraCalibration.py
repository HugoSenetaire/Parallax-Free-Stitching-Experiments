import numpy as np
import cv2
import glob
from argparse import ArgumentParser
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def cameraCalibration(path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(args["input"],"*.jpg"))
    images.extend(glob.glob(os.path.join(args["input"],"*.jpeg")))
    images.extend(glob.glob(os.path.join(args["input"],"*.png")))
    print(images)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = 127
        gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
        # Find the chess board corners
        # import pdb; pdb.set_trace()
        ret, corners = cv2.findChessboardCorners(gray, (7,9))
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    print(imgpoints,objpoints)
    # cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    parser = ArgumentParser(description='Tool for performing homographies')

    parser.add_argument('--input', required=True,
        help='Input path for a chekers')
   
    args = vars(parser.parse_args())
    print(args["input"])
    ret,mtx,dist,rvecs,tvecs = cameraCalibration(args["input"])

    print("ret",ret)
    print("mtx",mtx)
    print("dist",dist)
    print("rvecs",rvecs)
    print("tvecs",tvecs)
