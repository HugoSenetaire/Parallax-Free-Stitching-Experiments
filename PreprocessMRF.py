# import pymesh
# import pymesh.load_mesh
# from __future__ import d
import sklearn
import sklearn.cluster
import numpy as np
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement
import copy
# import tkinter as tkinter
# import tkinter as tk

from Tkinter import *
import Tkinter as tk
from tkFileDialog import askopenfilename
from sys import platform as sys_pf
from PIL import ImageTk
from PIL import Image

# Problem with MACOSX Use TKAGG as backend
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.tkagg as tkagg
import matplotlib.pyplot as plt
import tqdm
import os
import shapely
import math

import sklearn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

## UTILS 


# File :

def openImagesTxt(filename):
    X = []

    with open(filename,"r") as f :
        firstLine = True
        aux = []
        for line in f.readlines():
            if line.startswith("#") :
                continue
            if firstLine :
                firstLine = False
                aux.extend(line.split(" "))
            else :
                firstLine = True
                aux.append(np.array(line.split(" "),dtype = "f").reshape(-1,3))
                X.append(copy.deepcopy(aux))
                aux = []
    return X


def openPoints3D(filename):

    X = []

    with open(filename,"r") as f :
        aux = []
        for line in f.readlines():
            if line.startswith("#") :
                continue
            X.append(np.array(map(float,line.split(" "))))
    return np.array(X)

def openCamera3D(filename):
    X = []

    with open(filename,"r") as f :
        aux = []
        for line in f.readlines() :
            if line.startswith("#") :
                continue
            X.append(np.array(line.split(" ")))
    return np.array(X)

# Vector :


def orthogonalProduct(u,v):
    newVector = [
        u[1]*v[2]- u[2]*v[1],
        u[2]*v[0]- u[0]*v[2],
        u[0]*v[1]- u[1]*v[0],
        ]
    return newVector


def scalarProduct(u,v):
    return u[1]*v[1]+u[2]*v[2]+u[0]*v[0]


def projection(x,compo1,compo2):
    return [scalarProduct(x,compo1),scalarProduct(x,compo2)]


def normalize(x):
    return np.array(x) / np.linalg.norm(x)

def addVector(point,vector):
    x = point[0] + vector[0]
    y = point[1] + vector[1]
    z = point[2] + vector[2]
    return ([x,y,z])

#############===================#####################

## PREPROCESSING 1 : FIND PROJECTION PLANE 

def findPhotoPosition(data,colorPhoto = [0,255,0]):
    photoPos = []
    for point in data : 
        if np.all([point[3],point[4],point[5]]==colorPhoto):
            photoPos.append([point[0],point[1],point[2]])
    return photoPos



def automaticPlaneFinding(data):
    
    pca = PCA(n_components = 2)
    # print(photoPos)
    pca.fit(data)

    return pca.components_




def drawPlane(components, origin = [0,0,0]) :
   

    components = 5 * np.array(components)
    pt1 = addVector(origin,components[0])
    pt2 = addVector(origin,components[1])
    pt3 = addVector(addVector(origin,components[0]),components[1])
    return np.array([origin,pt1,pt2])




## PREPROCESSING 2 :  PICTURE SURFACE SELECTION




def drawAllPlt(scatterPoints,pos = []):
    fig = plt.figure(1)
    ax =  fig.add_subplot(1,1,1)
    ax.scatter(np.transpose(projectedVertices)[0]*100,np.transpose(projectedVertices)[1]*100,s=0.5)
    if len(pos)>0 :
        # print(pos)
        ax.scatter(np.transpose(pos)[0]*100,np.transpose(pos)[1]*100,s = 0.5)
    
    ax.axis('off')
    global coords
    coords = []

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)

        global coords
        coords.append((ix/100., iy/100.))

        if len(coords) == 20:
            fig.canvas.mpl_disconnect(cid)
        
        if len(coords)>=2 :
            plt.plot(np.transpose(coords)[0]*100, np.transpose(coords)[1]*100,marker = 'o',c='black')
            plt.show(1)
        else :
            plt.scatter(np.transpose(coords)[0]*100,np.transpose(coords)[1]*100,s =0.8)
            plt.show(1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(1)
    return coords



def deleteNoise(photoPos,points3D):
    norm = []
    points3D= np.array(points3D)
    for photo in photoPos :
        substract = np.subtract(np.array(points3D[:,1:]),np.array(photo))
        norm.append(np.linalg.norm(substract,axis =1))

    norm = np.transpose(np.array(norm))
    minNorm = np.min(norm,axis=1)
    meanMinNorm = np.mean(minNorm)
    stdMinNorm = np.std(minNorm)

    # for phot

    return None

def projection2D(point,pointLines):
    vector = np.array(pointLines)[1:]-np.array(pointLines)[:-1]

    proj= []
    dist = []
    possibleAux = []
    bestVec = None
    bestDist = float("inf")
    found = False
    for i in range(len(vector)) :
        vect = vector[i]
        aux = np.dot(point,vect) * vect + np.array(pointLines[i])
        if not (aux[0]>np.maximum(pointLines[i][0],pointLines[i+1][0]) or aux[0]<np.minimum(pointLines[i][0],pointLines[i+1][0])) :
            norm = np.linalg.norm(aux-point)
            found = True
            if norm<bestDist :
                bestVec = aux
                bestDist = norm

    if found :
        return bestVec 
    else :
        return [None,None]



def getMinNorm(points,pointLines):
    minNorm = []
    for point in points:
        norm = np.linalg.norm(np.array(pointLines)-np.array(point),axis = 1)
        minNorm.append(np.min(norm))
    

    return np.mean(minNorm),np.std(minNorm)



def projection2DFast(point,pointLines,meanNorm,stdNorm):
    vector = np.array(pointLines)[1:]-np.array(pointLines)[:-1]

    proj= []
    dist = []
    possibleAux = []
    bestVec = None
    bestDist = float("inf")
    found = False
    norm = np.linalg.norm(np.array(pointLines)-np.array(point),axis = 1)
    
    if np.min(norm)>meanNorm+1.0*stdNorm:
        return [None,None]
    indice = np.argmin(norm)
    
    for i in range(max(0,indice-2),min(len(pointLines)-1,indice+2)) :

        vect = vector[i]
        aux = np.dot(point,vect) * vect + np.array(pointLines[i])
        if not (aux[0]>np.maximum(pointLines[i][0],pointLines[i+1][0]) or aux[0]<np.minimum(pointLines[i][0],pointLines[i+1][0])) :
            norm = np.linalg.norm(aux-point)
            found = True
            if norm<bestDist :
                bestVec = aux
                bestDist = norm

    if found :
        return bestVec 
    else :
        for i in range(len(vector)) :
            vect = vector[i]
        aux = np.dot(point,vect) * vect + np.array(pointLines[i])
        if not (aux[0]>np.maximum(pointLines[i][0],pointLines[i+1][0]) or aux[0]<np.minimum(pointLines[i][0],pointLines[i+1][0])) :
            norm = np.linalg.norm(aux-point)
            found = True
            if norm<bestDist :
                bestVec = aux
                bestDist = norm

        if found :
            return bestVec 
        else :
            return [None,None]

        
def projectionEnsemble(projectedVertices,newPoints):
    mean,stdNorm = getMinNorm(projectedVertices,newPoints)
    doubleProjectedVertices = []
    distanceProjected = []
    # print(mean,stdNorm)
    compteur = 0
    for projPoint in tqdm.tqdm(projectedVertices) :        
        doubleProjectedVertices.append(projection2DFast(projPoint,newPoints,mean,stdNorm))
        
        # print(doubleProjectedVertices)
        if doubleProjectedVertices[-1][0]!= None:
            distanceProjected.append(np.linalg.norm(np.array(projPoint)-np.array(doubleProjectedVertices[-1])))
        else :
            compteur+=1
            distanceProjected.append(10000)
    print("Nb Points discarded : {}".format(compteur))
    return doubleProjectedVertices,np.array(distanceProjected)


def transform2dto3d(points2d,points3d,components2d,lastcomponent):
    convertedPoints= []
    count = 0
    for point2d in tqdm.tqdm(points2d) :
        if point2d[1]==None or point2d[2]==None :
            convertedPoints.append([point2d[0],None,None,None])
            count+=1
        else :
            index3d = np.where(points3d[:,0]==point2d[0])[0][0]

            new3D = point2d[1]*components2d[0]+point2d[2]*components2d[1]+np.dot(points3d[index3d,1:4],lastcomponent)*lastcomponent
            convertedPoints.append(np.concatenate(([point2d[0]],new3D),axis = 0))
    return convertedPoints




def createSurfaceImage(pointsLine,points3d,components2d,lastcomponent,nbline=2000):
    yComponent3D = []
    for i in tqdm.tqdm(range(len(points3d))):
        if points3d[i][2]!= None :
            yComponent3D.append(np.dot(np.array(points3d)[i,1:4],lastcomponent))
    print(np.shape(yComponent3D))
    maxY = np.max(yComponent3D)
    minY = np.min(yComponent3D)
    # print(maxY)
    # print(minY)


    step = (maxY-minY)/nbline
    surface = []
    for k in tqdm.tqdm(range(nbline+1)):
        for i in range(len(pointsLine)):
            surface.append(np.array(pointsLine[i][0]*components2d[0]+pointsLine[i][1]*components2d[1]+(minY + k*step)*lastcomponent))
    return surface




def getRotationFronQuaternion(a,b,c,d):
 
    # t1 = [
    #     [w,z,-y,x],
    #     [-z,w,x,y],
    #     [y,-x,w,z],
    #     [-x,-y,-z,w],
    # ]
    # t2 = [
    #     [w,z,-y,-x],
    #     [-z,w,x,-y],
    #     [y,-x,w,-z],
    #     [x,y,z,w],
    # ]  

    R = [
        [1-2*c**2-2*d**2,2*b*c - 2*a*d,2*b*d + 2*a*c],
        [2*b*c+2*a*d,1-2*b**2-2*d**2,2*c*d-2*a*b],
        [2*b*d-2*a*c,2*c*d+2*a*b,1-2*b**2-2*c**2],
    ]

    return np.array(R)


def surface3DtoImage(surfacePoint, features,camera):
    listFinal = []
    for k in range(len(features)) :
        cameraFocal = camera[:,4][0]
        imageFeature = features[k]
        rotationMatrix = getRotationFronQuaternion(float(imageFeature[1]),float(imageFeature[2]),float(imageFeature[3]),float(imageFeature[4]))
        translationMatrix = np.array([float(imageFeature[5]),float(imageFeature[6]),float(imageFeature[7])])
        K = np.array([[ 2.02029468e+05,  0.00000000e+00, -1.35724947e+03], \
       [ 0.00000000e+00,  5.61027217e+04,  1.38577670e+03],\
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        K = np.array(K)

        finalPoint = []
        for point in projected3D :
            if point[1] != None :
                vect = np.dot(K,np.concatenate((np.dot(rotationMatrix,point[1:])+translationMatrix,[1])))
                finalPoint.append(np.concatenate(([point[0]],vect))) 

        finalPoint =  np.array(finalPoint)
        listFinal.append(copy.deepcopy(finalPoint))


    return listFinal


import cv2

# def projectionInPicture(surfacePoint,features,camera,inputDir = "DESKCOLMAP/images"):
#        listImage = []
#     listFeature = []
#     listDistance = []
#     cameraFocal = camera[:,4]
#     cameraPrincipalPointX = camera[:,5]
#     cameraPrincipalPointY = camera[:,6]
#     cameraSkew = camera[:,7]
#     for imageFeature in tqdm.tqdm(features) :
#         focal = float(cameraFocal[int(imageFeature[8])-1])
#         principalPointx = float(cameraPrincipalPointX[int(imageFeature[8])-1])
#         principalPointy = float(cameraPrincipalPointY[int(imageFeature[8])-1])
#         skew = 0
#         rotationMatrix = getRotationFronQuaternion(float(imageFeature[1]),float(imageFeature[2]),float(imageFeature[3]),float(imageFeature[4]))
#         translationMatrix = np.array([float(imageFeature[5]),float(imageFeature[6]),float(imageFeature[7])])
#         K = [
#             [float(focal),skew,principalPointx],
#             [0,float(focal),principalPointy],
#             [0,0,1],
#         ]



#         surfaceInPicture = []
#         surfaceCentered = []
#         for line in surfacePoint :
#             surfaceInPictureLine = []
#             surfaceCenteredLine = []
#             for point in line :
#                 vect = np.dot(K,(np.dot(rotationMatrix,point)+translationMatrix))
#                 surfaceCenteredLine.append(np.dot(rotationMatrix,point)+translationMatrix)
#                 vect = vect/float(vect[2])
#                 vect = vect[:2]
#                 surfaceInPictureLine.append(vect)
#             surfaceInPicture.append(copy.deepcopy(surfaceInPictureLine)) # Coordonnees de la surface dans l'image
#             surfaceCentered.append(copy.deepcopy(surfaceCenteredLine)) # Coordonnees de la surface dans le referentiel de l'appareil
        

def reconstructionSurfaceFast(surfacePoint,features,camera,inputDir = "DESKCOLMAP/"):
    listImage = []
    listFeature = []
    listDistance = []
    cameraFocal = camera[:,4]
    print(len(cameraFocal))
    print("nb CAM:",len(cameraFocal))
    cameraPrincipalPointX  = camera[:,5]
    cameraPrincipalPointY = camera[:,6]
    cameraSkew = camera[:,7]
    for imageFeature in tqdm.tqdm(features) :
        focal = float(cameraFocal[int(imageFeature[8])-1])
        principalPointx = float(cameraPrincipalPointX[int(imageFeature[8])-1])
        principalPointy = float(cameraPrincipalPointY[int(imageFeature[8])-1])
        skew = 0
        rotationMatrix = getRotationFronQuaternion(float(imageFeature[1]),float(imageFeature[2]),float(imageFeature[3]),float(imageFeature[4]))
        translationMatrix = np.array([float(imageFeature[5]),float(imageFeature[6]),float(imageFeature[7])])
        K = [
            [float(focal),skew,principalPointx],
            [0,float(focal),principalPointy],
            [0,0,1],
        ]



        surfaceInPicture = []
        surfaceCentered = []
        for line in surfacePoint :
            surfaceInPictureLine = []
            surfaceCenteredLine = []
            for point in line :
                vect = np.dot(K,(np.dot(rotationMatrix,point)+translationMatrix))
                surfaceCenteredLine.append(np.dot(rotationMatrix,point)+translationMatrix)
                vect = vect/float(vect[2])
                vect = vect[:2]
                surfaceInPictureLine.append(vect)
            surfaceInPicture.append(copy.deepcopy(surfaceInPictureLine)) # Coordonnees de la surface dans l'image
            surfaceCentered.append(copy.deepcopy(surfaceCenteredLine)) # Coordonnees de la surface dans le referentiel de l'appareil
        

        surfaceInPicture = np.array(surfaceInPicture)
        imagePath = os.path.join(inputDir,"images",imageFeature[-2].strip("\n"))
        numpyImage = np.array(cv2.imread(imagePath))
        shapeImage = np.shape(numpyImage)
        sortieImage = np.zeros((np.shape(surfacePoint)[0],np.shape(surfacePoint)[1],3))
        valueFeature = -np.ones((np.shape(surfacePoint)[0],np.shape(surfacePoint)[1]))

      
       
        print(shapeImage)
        test = - np.ones((shapeImage[0],shapeImage[1]))
        points2d = np.reshape(imageFeature[-1],(-1,3)).astype("int")
        for point in points2d :
            test[point[1]-5:point[1]+5,point[0]-5:point[0]+5]= point[2]



        compteur = 0
        featuresTaken = [-1]
        distance = np.ones((np.shape(surfacePoint)[0],np.shape(surfacePoint)[1]))*float("inf")
        for line in range(len(surfaceInPicture)):
            for cols in range(len(surfaceInPicture[line])):
                x = surfaceInPicture[line][cols][0]
                y = surfaceInPicture[line][cols][1]
                if x>0 and x<shapeImage[1] \
                    and y>0 and y<shapeImage[0] :
                    compteur +=1
                    
                    if not test[int(y),int(x)] in featuresTaken :
                        valueFeature[line,cols] = test[int(y),int(x)]
                        featuresTaken.append(test[int(y),int(x)])
             
                    # print(surfaceCentered)
                    distance[line,cols] = np.linalg.norm(surfaceCentered[line][cols])
                    for k in range(3):
                        sortieImage[line,cols,k] = numpyImage[int(y),int(x),k] 
                        
        print("{} points were put on the surface".format(compteur))
        listImage.append(copy.deepcopy(sortieImage))
        listFeature.append(copy.deepcopy(valueFeature))
        listDistance.append(copy.deepcopy(distance))
    return listImage,listFeature,listDistance
        

                

def getPhotoPos(features):
    photoPos = []

    for imageFeature in features :
        # focal = cameraFocal[int(imageFeature[8])-1]
        rotationMatrix = getRotationFronQuaternion(float(imageFeature[1]),float(imageFeature[2]),float(imageFeature[3]),float(imageFeature[4]))
        translationMatrix = np.array([float(imageFeature[5]),float(imageFeature[6]),float(imageFeature[7])])
        photoPos.append(np.dot(-np.transpose(rotationMatrix),translationMatrix))
        # photoPos.append(translationMatrix)


    return photoPos


def findAxes(img):
    fig = plt.figure(1,figsize = (20,20))
    ax =  fig.add_subplot(1,1,1)
    ax.imshow(img)
    ax.axis('on')

    global coords
    coords = []
    

    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print 'x = %d, y = %d'%(
            ix, iy)

        global coords
        coords.append((ix, iy))

        plt.scatter(np.transpose(coords)[0],np.transpose(coords)[1])
        plt.show(1)

        if len(coords)==2 :
            plt.close()
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show(1)

    # print(coords)
    return coords



def getBestIndex(point,pointIndexList):
    bestDist = float("inf")
    index = -1
    # print("getBestIndex",point,pointIndexList[0])
    for pointIndex in pointIndexList :
        
        dist = np.linalg.norm(np.array(point)-pointIndex[0:2])
        if dist<bestDist and pointIndex[2]>=0 :
            bestDist = dist
            index = pointIndex[2]

    return index




def componentSelectionV2(features,point3d,camera,inputDir ):
    imageInit = -1
    indexX = []
    indexY = []
    coordsX = []
    coordsY = []
    while len(indexX)<2 or len(indexY)<2 : 
        imageInit = (imageInit + 1)%len(features)

        image = features[imageInit]
        rotationMatrix = getRotationFronQuaternion(float(image[1]),float(image[2]),float(image[3]),float(image[4]))
        translationMatrix = np.array([float(image[5]),float(image[6]),float(image[7])])
        newImgIndexes = []
        points2d = np.reshape(image[-1],(-1,3))
    

        point3d = np.array(point3d)
        # print(type(point3d[0]),point3d)
        cameraFocal = camera[:,4]
        focal = cameraFocal[int(image[8])-1]


        imagePath = os.path.join(inputDir,image[-2].strip("\n"))
        numpyImage = np.array(cv2.imread(imagePath))


        for point in points2d :
            if point[-1]>=0. :
                numpyImage[int(point[1])-5:int(point[1])+5,int(point[0])-5:int(point[0])+5,:] = np.array([255,255,255])



        
        if len(indexX)<2 :
            print("Find the X axis on the image")
            coordsX = findAxes(numpyImage)
            if len(coordsX)>=2 :
                indexX = [getBestIndex(coordsX[0],points2d),getBestIndex(coordsX[1],points2d)]
                print("indexX",indexX)
        if len(indexY)<2 :
            print("Find the Y axis on the image")
            coordsY = findAxes(numpyImage)
            if len(coordsY)>=2 :
                indexY = [getBestIndex(coordsY[0],points2d),getBestIndex(coordsY[1],points2d)]
                print("indexY",indexY)
    
    

    # A raccourcir :
    for i in range(len(point3d)) :
        if point3d[i][0] == int(indexX[0]):
            i1 = i
        elif point3d[i][0]== int(indexX[1]) :
            i2 = i

        if point3d[i][0] == int(indexY[0]):
            i3 = i
        elif point3d[i][0]== int(indexY[1]) :
            i4 = i 

    vectX = point3d[i2][1:4]-point3d[i1][1:4]
    vectY = point3d[i3][1:4]-point3d[i4][1:4]
    
    
    z = np.cross(vectX,vectY)
    x = np.cross(vectY,z)
    y = np.cross(z,x)


    x = normalize(x)
    z = normalize(z)
    y = normalize(y)
    

    return x,y,z

def findTranslation(feature1,feature2,commonFeatures):
    coordinates1 = np.zeros((len(commonFeatures),2))
    coordinates2 = np.zeros((len(commonFeatures),2))
    for i in range(len(feature1)) :
        for j in range(len(feature1[i])) :
            if feature1[i][j]!=-1 and feature1[i][j] in commonFeatures :
                place = np.where(feature1[i,j]==commonFeatures)[0]
                # print(place,commonFeatures[place],feature1[i,j])
                coordinates1[place]= [i,j]
            if feature2[i][j]!=-1 and feature2[i][j] in commonFeatures :
                place = np.where(feature2[i,j]==commonFeatures)[0]
                # print(place,commonFeatures[place],feature2[i,j])
                coordinates2[place]= [i,j]
    # print("GO")
    # for i in range(len(coordinates1)):
        # print(coordinates1[i],coordinates2[i])
        # print(feature1[int(coordinates1[i][0]),int(coordinates1[i][1])],feature2[int(coordinates2[i][0]),int(coordinates2[i][1])])

    aux1 = coordinates1[0].astype(int)
    aux2 = coordinates2[0].astype(int)
    translationVectors = (np.array(coordinates1)-np.array(coordinates2))


    # plt.scatter(np.transpose(translationVectors)[0],np.transpose(translationVectors)[1])
    # plt.show()
    # x = np.transpose(coordinates2)[0]
    # y = np.transpose(coordinates2)[1]
    # dx = x + np.transpose(translationVectors)[0]
    # dy = y + np.transpose(translationVectors)[1]
    # if True : 
        # return x,y,dx,dy
    # print(x,dx)
    # cmap = plt.cm.jet
    # for i in range(len(x)):
        # plt.arrow(x[i],y[i],dx[i],dy[i])
    # plt.show()
                # 
    # np.mean(translationVectors,axis = 0)
    # print("test")
    return(np.mean(translationVectors,axis = 0))
    


def translateImage(translationVector,image,distanceProjected2D):
    newImage = np.zeros(np.shape(image))









    return False



def cleanFeatures(features,distance):
    distanceMean= np.mean(distance[:,1])
    print(np.shape(features))
    for featureListIndex in tqdm.tqdm(range(len(features))) :
        featureList = features[featureListIndex]
        indices = np.where(featureList!=-1)
        # print(indices)
        properFeature = featureList[indices]
        for featureIndex in range(len(properFeature)) :
            feature = properFeature[featureIndex]

            if distance[np.where(distance[:,0]==feature)][0,1]>  0.01*distanceMean:
                # print(np.shape(features),featureListIndex,indices[0][featureIndex],indices[1][featureIndex])
                # print(features[0,0,450])
                features[featureListIndex][indices[0][featureIndex]][indices[1][featureIndex]] = -1


    return features



def alignSurfacePicture(images,features):
    alignedImages = [0]

    imagesTotal = []
    featuresTotal = []

    for indexImage in range(0,len(features)) :
        imagesTotal.append(images[indexImage])
        featuresTotal.append(features[indexImage][np.where(features[indexImage]>0)])
        currentImageFeatures = features[indexImage]


    # print(featuresTotal[0],featuresTotal[1])

    commonFeaturesNumber = np.zeros((len(features),len(features)))
    for indexImage in range(0,len(features)):
        for indexImage2 in range(indexImage+1,len(features)):
            commonFeaturesNumber[indexImage,indexImage2] = len(np.intersect1d(featuresTotal[indexImage],featuresTotal[indexImage2]))
            commonFeaturesNumber[indexImage2,indexImage] = len(np.intersect1d(featuresTotal[indexImage],featuresTotal[indexImage2]))

    print(commonFeaturesNumber)
    # First Feature Matching :
    index2List = []
    maxValue = 0
    for index1 in range(len(features)):
        index2 = np.argmax(commonFeaturesNumber[index1])
        index2List.append(index2)
        maxValue = np.max(commonFeaturesNumber[index1])
    
    index1 = np.argmax(maxValue)
    index2 = index2List[index1]


    listIndex = [index1,index2]
    # print("Common Feature Index",commonFeaturesNumber[index1,index2])
    # x,y,dx,dy = findTranslation(features[index1],features[index2],np.intersect1d(featuresTotal[index1],featuresTotal[index2]))
    translationVector = findTranslation(features[index1],features[index2],np.intersect1d(featuresTotal[index1],featuresTotal[index2]))
    translateImage(translationVector,image)


    # figure,(ax,ay)= plt.subplots(1,2,figsize = (20,10))
    # figure.size = (20,10)
    # ay.imshow(images[index1])
    # ax.imshow(images[index2])
    
    # for i in range(0,len(x),3):
    #     ax.scatter(y[i],x[i])
    #     ax.arrow(y[i],x[i],dy[i],dx[i])

    
    # plt.show(figure)


    # figure2 = plt.figure(1,figsize=(20,10))
    # ay = figure2.add_subplot(111)
    # ay.imshow(images[index1])
    # plt.show(figure2)

    # raw_input()
    # commonFeaturesNumber[index1,index2]=0
    # commonFeaturesNumber[index2,index1]=0


    
    





    # Ransac for best translation :



    # Application of the translation :




    return False

import random

            

## PIPELINE

if __name__ == "__main__":
    outputPath = "outputVideoBlur"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    inputPath = "FrameFromVideoNotBlur"


    camera = openCamera3D(os.path.join(inputPath,"cameras.txt"))
    cameraSize = camera[:,2:4]
    cameraFocal = camera[:,4]

    # # nbline = 3000 
    # # nbcols = 10000

    nbline = 1000
    nbcols = 4000

    # nbline = 500
    # nbcols = 1000


    features = openImagesTxt(os.path.join(inputPath,"images.txt"))
    vertices3D = openPoints3D(os.path.join(inputPath,"points3D.txt"))

    image0 = features[0]
    nbImage = np.shape(features)[0]

    print("The scene was taken with {} images".format(nbImage))
    


    photoPos = getPhotoPos(features)


    rotationMatrix = getRotationFronQuaternion(float(image0[1]),float(image0[2]),float(image0[3]),float(image0[4]))
    transpositionMatrix = np.array(photoPos[0])
    # photoPos = np.dot(rotationMatrix,transpositionMatrix)

    a = []
    aux = []
    for vertices in vertices3D :
        aux.append([vertices[0],float(vertices[1]),float(vertices[2]),float(vertices[3])])
        a.append([float(vertices[1]),float(vertices[2]),float(vertices[3])])
    components = automaticPlaneFinding(np.array(aux)[:,1:])
    a = np.transpose(a)
    # deleteNoise(photoPos,aux)

    # # Vectors :
    x = components[0]
    y = -components[1]
    z = np.cross(x,y)
    x = np.cross(y,z)
    z = normalize(z)
    x = normalize(x)
    y = normalize(y)
    print(x,y,z)
    # x,y,z = componentSelectionV2(features,vertices3D,camera,os.path.join(inputPath,"IMAGES"))
    print(x,y,z)
    
    listePoint_x = []
    listePoint_y = []
    listePoint_z = []
    for k in range(0,100,10):
        listePoint_x.append(k/10. * np.array(x))
        listePoint_y.append(k/10. * np.array(y))
        listePoint_z.append(k/10. * np.array(z))
        
    

    listePoint_x= np.transpose(listePoint_x)
    listePoint_y= np.transpose(listePoint_y)
    listePoint_z= np.transpose(listePoint_z)
    figure = plt.figure()
    ax = figure.add_subplot(111,projection = '3d')

   
    ax.scatter(listePoint_x[0],listePoint_x[1],listePoint_x[2],c = "black")
    ax.scatter(listePoint_y[0],listePoint_y[1],listePoint_y[2],c = "red")
    ax.scatter(listePoint_z[0],listePoint_z[1],listePoint_z[2],c = "blue")
    ax.scatter(a[0,::10],a[1,::10],a[2,::10])
    photoPos = np.transpose(photoPos)
    ax.scatter(photoPos[0],photoPos[1],photoPos[2])
    photoPos = np.transpose(photoPos)
    plt.show()

    # Projection in 2d zx plane
    projectedVertices = []
    for vertex in aux :
        projectedVertices.append(projection(vertex[1:],x,z))

    projectedPhotoPos = []
    for pos in photoPos:
        projectedPhotoPos.append(projection(pos,x,z))


    # Draw Line
    points =drawAllPlt(projectedVertices,projectedPhotoPos)



    # Get Poly
    coef = np.polyfit(np.transpose(points)[0],np.transpose(points)[1],3)
    p = np.poly1d(coef)
    init = points[0][0]
    end = points[-1][0]
    
    step = (end-init)/nbcols
    newPoints = []
    for k in range(nbcols+1):
        newPoints.append(np.array([init+step*k,p(init+step*k)]))


    # Projection on poly line :
    print("Projection on poly line")
    doubleProjectedVertices,distanceProjected2D = projectionEnsemble(projectedVertices,newPoints)
    print(np.shape(distanceProjected2D))
    projected = np.concatenate((np.array(aux)[:,0].reshape(-1,1),doubleProjectedVertices),axis = 1)
    distanceProjected2D = np.concatenate((np.array(aux)[:,0].reshape(-1,1),distanceProjected2D.reshape(-1,1)),axis=1)
    


    # Go back to 3D, creation of the surface
    print("GO back to 3D")
    projected3D = transform2dto3d(projected,np.array(aux),np.array([x,z]),y)
    surface = createSurfaceImage(newPoints,projected3D,np.array([x,z]),y,nbline = nbline)
    transposedSurface = np.transpose(surface)
    transposed3D = np.transpose(np.array(projected3D)[:,1:4])



    # figure = plt.figure()
    # ax = figure.add_subplot(111,projection = '3d')

   
    # ax.scatter(listePoint_x[0],listePoint_x[1],listePoint_x[2],c = "black")
    # ax.scatter(listePoint_y[0],listePoint_y[1],listePoint_y[2],c = "red")
    # ax.scatter(listePoint_z[0],listePoint_z[1],listePoint_z[2],c = "blue")
    # ax.scatter(a[0,::10],a[1,::10],a[2,::10])
    # ax.scatter(transposedSurface[0,::10],transposedSurface[1,::10],transposedSurface[2,::10])
    # photoPos = np.transpose(photoPos)
    # ax.scatter(photoPos[0],photoPos[1],photoPos[2])
    # photoPos = np.transpose(photoPos)
    # plt.show()

    test = [] 
    for point in projected3D :
        if point[1]!=None :
            test.append([point[1],point[2],point[3]])

    test = np.transpose(np.array(test))

    
    # Faire une representation 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  


    
    
    # Trouver l'equation surface dans le plan camera : Pas necessaire
    print("reconstruction and Save")
    
    surface = np.array(surface).reshape(nbline+1,-1,3)
    images,features,distance = reconstructionSurfaceFast(surface,features,camera,inputDir=inputPath)
    compteur = 0
    for img in range(len(images)) :
        cv2.imwrite(os.path.join(outputPath,"img{}.jpg".format(compteur)),images[img])
        compteur+=1
   
    import cPickle as pickle
    picklePath = os.path.join(outputPath,"pickleAux")
    if not os.path.exists(picklePath):
        os.makedirs(picklePath)
    with open(os.path.join(picklePath,"features.pck"),"wb") as f :
        pickle.dump(features,f)

    with open(os.path.join(picklePath,"distance.pck"),"wb") as f :
        pickle.dump(distance,f)

    
    with open(os.path.join(picklePath,"distanceProjected.pck"),"wb") as f :
        pickle.dump(distanceProjected2D,f)



    # # ShortCut
    # import cPickle as pickle
    # images = []
    # for compteur in range(15) :
    #     images.append(np.array(cv2.imread(os.path.join(outputPath,"img{}.jpg".format(compteur)))))

    # # print(np.shape(images),np.shape(features))
    # with open("pickleAux/features.pck",'rb') as f:
    #     features = pickle.load(f)
    # with open("pickleAux/distanceProjected.pck","rb") as f:
    #     distanceProjected2D = pickle.load(f)



    # # # compteur = 0
    # # # if not os.path.exists("output/"):
    # # #     os.makedirs("output/")
    # # #     print("test")

  
    # # # print(np.shape(features))
    # features = np.array(features)
    # features = cleanFeatures(features,distanceProjected2D)
    # alignSurfacePicture(images,features)

    
    # print("Test",surfaceImage)
    # finalPoint = []
    # for point in projected3D :
    #     if point[1] != None :
    #         vect = np.dot(K,np.concatenate((np.dot(rotationMatrix,point[1:])+transpositionMatrix,[1])))
    #         finalPoint.append(np.concatenate(([point[0]],vect))) 

    # finalPoint =  np.array(finalPoint)
    # test = surface3DtoImage(projected3D,features,camera)
    # print(np.shape(test))
    # for image in test :
    #     print(np.max(image[:,1])-np.min(image[:,1]))
    #     print(np.max(image[:,2])-np.min(image[:,2]))
    #     print(np.max(image[:,1]),np.min(image[:,1]))
    #     print(np.max(image[:,2]),np.min(image[:,2]))






    


















    


