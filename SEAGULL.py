from PIL import *
import numpy as np
import numpy
import os
import cv2
from matplotlib.pyplot import imshow
import cPickle as pickle
from math import isinf,isnan
import glob
import copy
import tqdm
import scipy
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
import skimage
import matplotlib.pyplot as plt
import math

numSegment = 100
globalWidthStep = 20
globalHeightStep = 20
epsilon = 1e-2
sigma_m = 10
lambdaBig = 1.5
lambdaSmall = 0.1
minDistThreshold = 20
sampleSegmentNumber = 10

lambda1 = 5#5
lambda2 = 1#1
lambda3 = 0 #10

##



### ====================================================================================
### DETECTING FEATURES
### ====================================================================================

def detect_and_describe(image, width = 600):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detection keyponts
    surf = cv2.xfeatures2d.SURF_create(nOctaves=2, upright=True)
    kps = surf.detect(gray)
    if not kps:
        surf = cv2.xfeatures2d.SURF_create(upright=True)
        kps = surf.detect(gray)

    # Compute descriptors
    daisy = cv2.xfeatures2d.DAISY_create()
    kpscv, descriptors = daisy.compute(gray, kps)
    kps = np.float32([kp.pt for kp in kpscv])

    return kps, descriptors,kpscv

def featureMatching(featuresA,featuresB,imageA,imageB):
    
    
    kpsA, descriptorA, kpscvA = featuresA
    kpsB, descriptorB, kpscvB = featuresB
    # kpsA, descriptorA, SA = detect_and_describe(featuresA)
    # kpsB, descriptorB, SB = detect_and_describe(featuresB)
    matcher = cv2.DescriptorMatcher_create("FlannBased")
    ratio=0.7
    matches = matcher.knnMatch(descriptorA, descriptorB, 2) 
    matchescv = copy.deepcopy(matches)
    matches = filter(lambda m: len(m) == 2 and m[0].distance < m[1].distance * ratio, matches)
    
    
    # res = cv2.drawMatchesKnn(imageA,kpscvA,imageB,kpscvB,matches,None) 
    # plt.imshow(res)
    # plt.show()
    matchesAux = []
    matches = [(m[0].trainIdx, m[0].queryIdx) for m in matches]
    ptsA = np.array([kpsA[k] for (_, k) in matches])
    ptsB = np.array([kpsB[k] for (k, _) in matches])

    return ptsA,ptsB,matches



from skimage.segmentation import slic
def segmentation(image,numSegment = numSegment):
    segments = skimage.segmentation.slic(image,n_segments=numSegment)
    return segments 




def getLabeling(ptsA,ptsB,segmentsB):
    dic = {}
    found = [False]*numSegment


    for i in range(len(ptsA)):
        # print(ptsA[i],ptsB[i])
        
        
        if not found[segmentsB[ptsB[i][1],ptsB[i][0]]]:
            dic[segmentsB[ptsB[i][1],ptsB[i][0]]] = [[ptsA[i]],[ptsB[i]]]
            found[segmentsB[ptsB[i][1],ptsB[i][0]]] = True
        else :
            dic[segmentsB[ptsB[i][1],ptsB[i][0]]][0].append(ptsA[i])
            dic[segmentsB[ptsB[i][1],ptsB[i][0]]][1].append(ptsB[i])

    return dic

def is_on_edge(point,shape):
    if point[0] == shape[0]-1 or point[1]==shape[1]-1 or point[0]==0 or point[1]==0:
        return True
    else :
        return False

def get_neighbouring_label(seg):
    """ MAY GET FASTER """
    # neighbouring = {}
    # for i in range(numSegment):
    #     neighbouring[i] = np.array([i])
    # grad = np.linalg.norm(np.gradient(seg),axis=0)
    # # plt.imshow(np.where(grad!=0,250,0))
    # # plt.show()
    # listeGrad = np.transpose(np.where(grad!=0))
    # for pointindex in range(0,len(listeGrad),len(listeGrad)/1000) :
    #     point = listeGrad[pointindex]
    #     if not is_on_edge(point,np.shape(seg)):
    #         liste = [seg[point[0]-1,point[1]],seg[point[0]+1,point[1]],seg[point[0],point[1]+1],seg[point[0],point[1]-1]]
            
    #         segNeighbour = np.unique(liste)
            
    #         for i in range(numSegment) :
    #             # print(segNeighbour,neighbouring[i])
    #             neighbouring[i]= np.union1d(neighbouring[i],segNeighbour)
    neighbouring = {}
    for i in range(numSegment):
        neighbouring[i] = np.array([i])
    for i in tqdm.tqdm(range(1,len(seg)-1)):
        for j in range(1,len(seg[i])-1):
            neighbouring[seg[i,j]] = np.union1d(neighbouring[seg[i,j]],[seg[i+1,j],seg[i,j+1]])
            neighbouring[seg[i+1,j]] = np.union1d(neighbouring[seg[i+1,j]],[seg[i,j],seg[i,j+1]])
            neighbouring[seg[i,j+1]] = np.union1d(neighbouring[seg[i+1,j]],[seg[i+1,j],seg[i,j]])
    return neighbouring

def refine_to_label(dic):
    for label in dic.keys():
            # print("BEFORE",dic[label][0])
            M, mask = cv2.findHomography(np.array(dic[label][0]),np.array(dic[label][1]), cv2.RANSAC,5.0)
            if len(np.where(mask.reshape(-1))[0])<=2:
                del dic[label]
            else :
                dic[label][0] = np.array(dic[label][0])[np.where(mask.reshape(-1))]
                dic[label][1] = np.array(dic[label][1])[np.where(mask.reshape(-1))]
            # print("WfTER",dic[label][0])
        
    labelValues = list(dic.keys())
    labelValues = sorted(labelValues,key = lambda l : len(dic[l][0]),reverse= True)

    return dic,labelValues


def refine_to_group(dic,labelValues,neighboursMap):
    dicGroup = {}
    groupValue = [-1]*len(labelValues)
    print(len(groupValue))
    group =-1


    while -1 in groupValue:
        group+=1
        # print("WHERE",groupValue.index(-1))
        rootIndex = groupValue.index(-1)
        root = labelValues[rootIndex]
        groupValue[rootIndex] = group


        toTreat = list(neighboursMap[root])
        treated = [False]*len(labelValues)
        treated[rootIndex]=True

        dicGroup[group] = dic[root]
        while len(toTreat)>0 :
            currentLabel = toTreat.pop(0)
            currentLabelIndex = labelValues.index(currentLabel)
            if groupValue[currentLabelIndex]==-1 and not treated[currentLabelIndex] :
                treated[currentLabelIndex]=True
                

                M, mask = cv2.findHomography(np.concatenate((dicGroup[group][0],dic[currentLabel][0]),axis=0)\
                    ,np.concatenate((dicGroup[group][1],dic[currentLabel][1]),axis=0), cv2.RANSAC,5.0)
                # print(mask)

                if np.all(mask.reshape(-1)): #or len(np.where(mask.reshape(-1))[0])>max(len(dicGroup[group][0])+0.5*len(dic[currentLabel][0]),0.5*len(dicGroup[group][0])+len(dic[currentLabel][0])):
                    # print("TROUVE",root,currentLabel)
                    groupValue[currentLabelIndex]= group
                    toTreat.extend(list(neighboursMap[currentLabel]))
                    dicGroup[group][0] = np.concatenate((dicGroup[group][0],dic[currentLabel][0]))[mask.reshape(-1)]
                    dicGroup[group][1] = np.concatenate((dicGroup[group][1],dic[currentLabel][1]))[mask.reshape(-1)]
    



    print(groupValue)
    for groupNumber in dicGroup.keys():
        if groupNumber not in groupValue:
            del dicGroup[groupNumber]
    print(len(dicGroup.keys()))


    return(dicGroup,groupValue)
    

def refine_to_super_group(dicGroup):
    groupValues = list(dicGroup.keys())
    groupValues = sorted(groupValues,key = lambda l : len(dicGroup[l][0]),reverse= True)

    dicSuperGroup = {}
    superGroupValue = [-1]*len(groupValues)
    
    superGroup =-1
    for i in range(len(groupValues)-1):
        if superGroupValue[i] ==-1:
            superGroup+=1
            superGroupValue[i] = superGroup
            dicSuperGroup[superGroup]=dicGroup[i]

            for j in range(i+1,len(groupValues)):
                if superGroupValue[j]!=-1 :
                    M, mask = cv2.findHomography(np.concatenate((dicSuperGroup[superGroup][0],dicGroup[j][0]),axis=0)\
                    ,np.concatenate((dicSuperGroup[superGroup][1],dicGroup[j][1]),axis=0), cv2.RANSAC,5.0)
                    # if np.all(mask.reshape(-1)) :
                    if np.all(mask.reshape(-1)) or len(np.where(mask.reshape(-1))[0])>len(dicSuperGroup[superGroup][0])+0.5*len(dicGroup[j][0]) :
                        # print("TROUVE",root,currentLabel)


                        groupValue[currentLabelIndex]= group
                        toTreat.extend(list(neighboursMap[currentLabel]))
                        dicSuperGroup[superGroup][0] = np.concatenate((dicSuperGroup[superGroup][0],dicGroup[j][0]))[mask.reshape(-1)]
                        dicSuperGroup[superGroup][1] = np.concatenate((dicSuperGroup[superGroup][1],dicGroup[j][1]))[mask.reshape(-1)]


    return dicSuperGroup,superGroupValue


def refineGlobal(images,show=False):
    dicImages ={}
    for indexImage in range(len(images)-1):
        
        shape = np.shape(images[0])

        # Get Features
        features1 = detect_and_describe(images[indexImage])
        features2 = detect_and_describe(images[indexImage+1])
        ptsA,ptsB,matches = featureMatching(features1,features2,images[indexImage],images[indexImage+1])

        print(len(ptsA),len(ptsB))
        ptsA = ptsA.astype(int)
        ptsB = ptsB.astype(int)

        segB = segmentation(images[indexImage+1]) 
        dic = getLabeling(ptsA,ptsB,segB)

        dic,labelValues = refine_to_label(dic)

        neighboursMap = get_neighbouring_label(segB)
        for key in neighboursMap.keys():
            if key not in labelValues :
                del neighboursMap[key]
            else :
                neighboursMap[key] = np.intersect1d(neighboursMap[key],labelValues)

        
        dicGroup, groupValue = refine_to_group(dic,labelValues,neighboursMap)
        dicSuperGroup,superGroupValue = refine_to_super_group(dicGroup)
        
        if show :
            for key in dicSuperGroup.keys():

                res = cv2.drawMatchesKnn(images[indexImage],features1[2],images[indexImage+1],features2[2],[],None) 
                plt.imshow(res)
                for k in range(len(dicSuperGroup[key][0])) :
                    ptsA = dicSuperGroup[key][0][k]
                    ptsB = dicSuperGroup[key][1][k]
                    plt.plot([ptsA[0],ptsB[0]+shape[1]],[ptsA[1],ptsB[1]],linewidth = 1)
                plt.show()

        dicImages[indexImage,indexImage+1] = dicSuperGroup


    return dicImages


### ====================================================================================
### CREATION OF MATRICES
### ====================================================================================


### SIMPLE TOOLS :
def dico_to_array(dicSuperGroup):
    outArray1 = []
    outArray2 = []
    for k in dicSuperGroup.keys():
        for l in range(len(dicSuperGroup[k][0])):
            outArray1.append(dicSuperGroup[k][0][l])
            outArray2.append(dicSuperGroup[k][1][l])

    shape1 = np.shape(outArray1)
    assert(shape1[-1]==2) # points 2d
    
    return np.array(outArray1),np.array(outArray2)


### GRID CREATION :
def create_grid(image,heightStep=globalHeightStep,widthStep=globalWidthStep):
    """Create the grid for the target image
    NOT TESTED """

    height,width = np.shape(image)[:2]

    V = []
    for y in range(heightStep+1) :
        for x in range(widthStep+1):
            V.append(y*float(height-1)/(heightStep))
            V.append(x*float(width-1)/(widthStep))

    return np.array(V)



### FEATURE TERM :
def adaptive_feature_weight(distanceSeam,distanceReference):
    """ For eWch feature, create the weight associated, 
    NOT TESTED YET"""
    wFeature = []
    for i in range(len(distanceSeam)):
        if distanceSeam[i]<minDistThreshold:
            wFeature.append(lambdaBig*(np.exp(-(distanceReference[i]**2)/(2*sigma_m**2))+epsilon))
        else :
            wFeature.append(lambdaSmall*(np.exp(-(distanceReference[i]**2)/(2*sigma_m**2))+epsilon))

    return np.array(wFeature)


def find_corners(targetImage,location,heightStep=globalHeightStep,widthStep=globalWidthStep):
    """ For eWch location, find the indices of the four corners associated. 
    NOT TESTED : NEED TO CHECK IF LOCATION IS (X,Y) or (Y,X)
    NEED TO CHECK THE RESULT """

    height,width = np.shape(targetImage)[:2]
 
 
    yStep = float(height-1)/heightStep
    xStep = float(width-1)/widthStep

    yIndex = int(np.floor(location[1]/yStep))
    xIndex = int(np.floor(location[0]/xStep))

    
    index = 2*(yIndex*(widthStep+1)+xIndex)

    

    return [index,index+2,index+2*(widthStep+1),index+2*(widthStep+1)+2]

def fill_interpolation_matrice(corner,V,location,heightStep = globalHeightStep,widthStep=globalWidthStep):
    """ For eWch location, fill the matrix indices. 
    LOCATION IS X,Y
    NOT TESTED : NEED TO CHECK IF LOCATION IS (X,Y) or (Y,X)
    SHOULD BE X,Y for point and Y,X for V[CORNER]V[CORNER+1]
    NEED TO CHECK THE RESULT """
    
    

    W = np.zeros((2,len(V)))
    w1 = (V[corner[3]]-location[1])*(V[corner[3]+1]-location[0])
    w2 = -(V[corner[2]]-location[1])*(V[corner[2]+1]-location[0])
    w3 = -(V[corner[1]]-location[1])*(V[corner[1]+1]-location[0])
    w4 = (V[corner[0]]-location[1])*(V[corner[0]+1]-location[0])

    if len(np.where([w1,w2,w3,w4]==0))==2:
        if w1 ==0 and w2 ==0 : 
            w1 = abs(V[corner[3]+1]-location[0])
            w2 = abs(V[corner[2]+1]-location[0])
            wtot = w1+w2

            W[0,corner[3]] = w1
            W[1,corner[3]+1] = w1

            W[0,corner[2]] = w2
            W[1,corner[2]+1] = w2

        elif w1 == 0 and w3 ==0 :
            w1 = abs(V[corner[3]]-location[1])
            w3 = abs(V[corner[1]]-location[1])
            wtot = w1+w3

            W[0,corner[3]] = w1
            W[1,corner[3]+1] = w1

            W[0,corner[1]] = w3
            W[1,corner[1]+1] = w3
        
        elif w2 == 0 and w4 ==0 :
            w2 = abs(V[corner[2]]-location[1])
            w4 = abs(V[corner[0]]-location[1])
            wtot = w2+w4

            W[0,corner[2]] = w2
            W[1,corner[2]+1] = w2

            W[0,corner[0]] = w4
            W[1,corner[0]+1] = w4

        elif w3 == 0 and w4 ==0 :
            w3 = abs(V[corner[1]]-location[1])
            w4 = abs(V[corner[0]]-location[1])

            W[0,corner[1]] = w3
            W[1,corner[1]+1] = w3

            W[0,corner[0]] = w4
            W[1,corner[0]+1] = w4
    
    elif len(np.where([w1,w2,w3,w4]==0)[0])>2 or len(np.where([w1,w2,w3,w4]==0)[0])==1 :
        print(len(np.where(np.array([w1,w2,w3,w4])==0)))
        print(w1,w2,w3,w4)
        raise Exception
    else :

        wtot = w1+w2+w3+w4

        W[0,corner[0]] = w1
        W[1,corner[0]+1] = w1

        W[0,corner[1]] = w2
        W[1,corner[1]+1] = w2

        W[0,corner[2]] = w3
        W[1,corner[2]+1] = w3

        W[0,corner[3]] = w4
        W[1,corner[3]+1] = w4


    W = W/wtot
    Waux = csr_matrix(W)

    return Waux


def create_interpolation_feature_matrices(targetImage,featuresTarget,V):
    """ find the interpolation matrix for every features
    LOCATION IS X,Y
    NOT TESTED YET : CHeck if coordinates with multiplication from matrix are equal
    """
    W_feature = []

    for location in featuresTarget :
            
            corners = find_corners(targetImage,location)
            
            assert(V[corners[0]]<=location[1]and V[corners[0]+1]<=location[0])
            assert(V[corners[1]]<=location[1]and V[corners[1]+1]>=location[0])
            assert(V[corners[2]]>=location[1]and V[corners[2]+1]<=location[0])
            assert(V[corners[3]]>=location[1]and V[corners[3]+1]>=location[0])
            W_feature.append(fill_interpolation_matrice(corners,V,location))
    # assert(1==0)
    return W_feature
            
           
            
 ### LOCAL TERM :


def calculate_UV(targetImage,Vinit,widthStep=globalWidthStep,heightStep=globalHeightStep) :
    """ U et V Toujours identique dans un sens puis dans l'autre 
    Verifie par deux calculs differents
    """



    height,width = np.shape(targetImage)[:2]
    heightLength = float(height-1)/heightStep
    widthLength = float(width-1)/widthStep
    UV = [
        heightLength**2/np.linalg.norm([heightLength,widthLength]),
        heightLength*widthLength/np.linalg.norm([heightLength,widthLength])
        ]
    print("UV",UV)
    return UV

def calculate_UV2(targetImage,V,widthStep=globalWidthStep,heightStep=globalHeightStep):
    """ U et V Toujours identique dans un sens puis dans l'autre 
    Verifie par deux calculs differents
    """
    W = np.zeros((2,len(V)))
    W[0,0] = 1
    W[1,1] = 1

    Wc = np.zeros((2,len(V)))
    Wc[0,2] = 1
    Wc[1,3] = 1
            
    Wb = np.zeros((2,len(V)))
    Wb[0,2*(widthStep+1)] = 1
    Wb[1,2*(widthStep+1)+1] = 1



    vectorU = np.matmul(Wc-Wb,V)
    vectorU = vectorU/(np.linalg.norm(vectorU)**2)
    vectorV = np.array([-vectorU[1],vectorU[0]])
    print("VECTORU,VECTORV",vectorU,vectorV)

    u = np.dot(vectorU,np.matmul(W-Wb,V))
    v = np.dot(vectorV,np.matmul(W-Wb,V))
    print("UV",[u,v])
    return [u,v]

    

    # UV = []
    # for i in range((widthStep+1)*heightStep):

    # for j in range(widthStep+1,(widthStep+1)*(heightStep+1)):

def create_local_preserving_matrices(targetImage,V,widthStep=globalWidthStep,heightStep=globalHeightStep,init=False) :
    """ Creer la matrice de preservation de structure 

    NOT TESTED YET : """


    UV = calculate_UV2(targetImage,V,widthStep=widthStep,heightStep=heightStep)


    Wtot = [] 
    u = UV[0]
    v = UV[1]
    # Premier triangle(base en haut a gauche)
    for y in range(heightStep):
        for x in range(widthStep):
            index = 2*(y*(widthStep+1)+x)
            indexB = 2*((y+1)*(widthStep+1)+x)
            indexC = 2*(y*(widthStep+1)+(x+1))
            
            # # index = 2*i
            W = np.zeros((2,len(V)))
            W[0,index] = 1 #ya sortie y
            W[1,index+1] = 1 #xa sortie x

            # print(np.matmul(W,V))

            W[0,indexB] =  -1+u#yb sortie y
            W[0,indexB+1] = -v #xb sortie y
            W[1,indexB] =  +v #yb sortie x
            W[1,indexB+1] = -1+u   #xb sortie x
           
            W[0,indexC] = -u #yc sortie y
            W[0,indexC+1] = v #xc sortie y
            W[1,indexC] = -v #yc sortie x
            W[1,indexC+1] = -u #xc sortie x



            if init and np.linalg.norm(np.matmul(W,V))>1e-3 :
                print("MATRICE W",csr_matrix(W))
                print("WHERE",np.where(W!=0))
                print("VALUE",W[np.where(W!=0)])
                print("SORTIE",np.matmul(W,V))
                raise Exception
            
            Wtot.append(csr_matrix(W))
    #Second triangle (Base en bas a droite)
    for y in range(1,heightStep+1):
        for x in range(1,widthStep+1):
            index = 2*(y*(widthStep+1)+x)
            indexB = 2*((y-1)*(widthStep+1)+x)
            indexC = 2*(y*(widthStep+1)+(x-1))
            # index = 2*i
            W = np.zeros((2,len(V)))
            W[0,index] = 1 #ya sortie y
            W[1,index+1] = 1 #xa sortie x

            W[0,indexB] =  -1+u#yb sortie y
            W[0,indexB+1] = -v #xb sortie y
            W[1,indexB] =  v #yb sortie x
            W[1,indexB+1] = -1+u   #xb sortie x

            W[0,indexC] = -u #yc sortie y
            W[0,indexC+1] = +v #xc sortie y
            W[1,indexC] = -v #yc sortie x
            W[1,indexC+1] = -u #xc sortie x

            if init and np.linalg.norm(np.matmul(W,V))>1e-3 :
                print("MATRICE W",csr_matrix(W))
                print("WHERE",np.where(W!=0))
                print("SORTIE",np.matmul(W,V))
                raise Exception
            
            Wtot.append(csr_matrix(W))

    return Wtot


### Create Curve Segments Term

def interpolation_curve_matrices(targetImage,points,V,heightStep=globalHeightStep,widthStep=globalWidthStep):
    """ find the interpolation matrix for every point in points
    NOT TESTED YET : CHeck if coordinates with multiplication from matrix are equal
    """
    W_feature = []
    for location in points:

            corners = find_corners(targetImage,location,heightStep=heightStep,widthStep=widthStep)
            aux = copy.deepcopy(fill_interpolation_matrice(corners,V,location,heightStep=heightStep,widthStep=widthStep))
            assert(np.shape(aux)==(2,len(V)))
            W_feature.append(aux)
    
    return W_feature

def create_curve_points(targetImage,V,showCurve = False) :
    """ CONTOUR CHECKED,
    MATRICES NOT CHECKED"""
    img = targetImage
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    contours = filter(lambda m: np.shape(m)[0]>30,contours)
    

    if showCurve :
        for contour in contours:
            print(np.shape(contour))
            print(contour)
        for i in range(len(contours)):
            aux = cv2.drawContours(img, contours, i, (0,255,0), 3)
            plt.imshow(aux)
            plt.show()
    
    contourMatrix = []
    for contour in contours :
        contourMatrix.append(interpolation_curve_matrices(targetImage,contour[0::sampleSegmentNumber,:,:].reshape(-1,2),V))
    
    
    return contourMatrix

def create_curve_matrices(targetImage,V,showCurve = False,init=False,widthStep=globalWidthStep,heightStep=globalHeightStep):
    """ Create Matrix associated with contours optimisation from contour points
    NOT TESTED  """


    contourMatrix = create_curve_points(targetImage,V,showCurve=showCurve)

    
    # print(contourMatrix)
    Wtot = []
    for contour in contourMatrix:
        

        vectorU = (contour[-1]-contour[0]).dot(V)
        
        vectorV = [vectorU[1],-vectorU[0]]
        
        vectorU = vectorU/(np.linalg.norm(vectorU)**2)
        vectorV = vectorV/(np.linalg.norm(vectorV)**2)

        Wb = contour[0].toarray()
        Wc = contour[-1].toarray()
     

        for i in range(1,len(contour)-1):
            W = np.zeros((2,len(V)))

            vectorVkey = (contour[i]-contour[0]).dot(V)
            Wkey = contour[i].toarray()
            

            u = np.dot(vectorU,vectorVkey)
            v = np.dot(vectorV,vectorVkey)
            Wdiff = Wc-Wb
            # print(Wdiff)
            W[0] = Wkey[0]-(Wb[0]+u*(Wc[0]-Wb[0])+v*(Wc[1]-Wb[1]))
            W[1] = Wkey[1]-(Wb[1]+u*(Wc[1]-Wb[1])-v*(Wc[0]-Wb[0]))


            if init and np.linalg.norm(np.matmul(W,V))>1e-3 :
                print("MATRICE W",csr_matrix(W))
                print("WHERE",np.where(W!=0))
                print("VALUE",W[np.where(W!=0)])
                print("SORTIE",np.matmul(W,V))
                raise Exception

            Wtot.append(csr_matrix(W))

    return Wtot


### Calculate Energy :
def energyFeature(W,V,featuresOrigin,weight):
    
    energy = 0
    for i in range(len(W)) :
        # energy+=weight[i]*np.linalg.norm(W[i].dot(V)-featuresOrigin[i])**2
        aux = [featuresOrigin[i][1],featuresOrigin[i][0]]
        energy+=weight[i]*(np.linalg.norm(W[i].dot(V)-aux)**2)

    return lambda1*energy


def energyLocal(W,V):
    energy = 0
    for i in range(len(W)):
        energy+=np.linalg.norm(W[i].dot(V))**2
    
    return lambda2*energy

def energyCurve(W,V):
    energy = 0
    for i in range(len(W)):
        energy+=np.linalg.norm(W[i].dot(V))**2
    return lambda3*energy

### ====================================================================================
### SEAMING
### ====================================================================================

def defineSeaming():
    return False





### ====================================================================================
### SOLVING
### ====================================================================================


### SOLVE FOR ONE ITERATION:

def solve(targetImage,featuresOrigin,featuresTarget):
    V =create_grid(targetImage)

  


    Wf = create_interpolation_feature_matrices(targetImage,featuresTarget,V)
    Wc = create_curve_matrices(targetImage,V,init=True)
    Wl = create_local_preserving_matrices(targetImage,V,init=True)
    distanceSeam = np.zeros((len(featuresOrigin)))
    features = []
    for i in range(len(Wf)):
        aux = Wf[i].dot(V)
        features.append([aux[1],aux[0]])

    features = np.array(features)
    distanceReference = np.linalg.norm(featuresOrigin-features,axis=1)
    adaptiveFeatureWeight=adaptive_feature_weight(distanceSeam,distanceReference)

    assert(len(featuresOrigin)==len(featuresTarget))
    
  
    def energy_total(V):
 
        Ef = energyFeature(Wf,V,featuresOrigin,adaptiveFeatureWeight)
        print("ENERGY FEATURE :",Ef)
        Ec = energyCurve(Wc,V)
        print("ENERGY CURVE :",Ec)
        El = energyLocal(Wl,V)
        print("ENERGY LOCAL :",El)
        print("ENERGY :",Ef+Ec+El)
        return Ef+Ec+El
    test = np.mean(featuresOrigin-featuresTarget,axis=0)
    # print(test)
    # print(energy_total(V))
    for i in range(len(V)):
        # print(V[i],test[(i+1)%2])
        V[i] = V[i]+test[(i+1)%2]
    # print(energy_total(V))
    def grad(V):
        auxF = []
        for i in range(len(distanceReference)):
            auxF.append(Wf[i].dot(V)-distanceReference[i])

        gradF = np.zeros(np.shape(V))
        for i in range(len(distanceReference)):
            # print(np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i])
            gradF +=np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i]
        # print("GRADF",gradF)


        gradC = np.zeros(np.shape(V))
        for i in range(len(Wc)):
            # print(np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V)))
            gradC+=np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V))
        print("GRADC",np.linalg.norm(gradC))

        gradL= np.zeros(np.shape(V))
        for i in range(len(Wl)):
            gradL+=np.matmul(np.transpose(Wl[i].toarray()),Wl[i].dot(V))
        print("GRADL",np.linalg.norm(gradL))
        

        return lambda1*gradF+lambda3*gradC+lambda2*gradL
    # grad(V)
    # res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy_total,V,approx_grad=True,iprint = 99 ))
    
    # energy_total(res[0])
    # print(np.linalg.norm(res[0]-V))

 


    visuGrid(V,targetImage)

def solve2(targetImage,featuresOrigin,featuresTarget):
    V =create_grid(targetImage)
    
    distanceReference = np.linalg.norm(featuresOrigin-featuresTarget,axis=1)
    print(distanceReference)
    distanceSeam = np.zeros(np.shape(distanceReference))

    adaptiveFeatureWeight=adaptive_feature_weight(distanceSeam,distanceReference)
    # print("WEIGHT",adaptiveFeatureWeight)
    Wf = create_interpolation_feature_matrices(targetImage,featuresTarget,V)
    Wc = create_curve_matrices(targetImage,V,init=True)
    Wl = create_local_preserving_matrices(targetImage,V,init=True)


    
    def energy_total(V):
        
        Ef = energyFeature(Wf,V,featuresOrigin,adaptiveFeatureWeight)
        print("ENERGY FEATURE :",Ef)
        Ec = energyCurve(Wc,V)
        print("ENERGY CURVE :",Ec)
        El = energyLocal(Wl,V)
        print("ENERGY LOCAL :",El)
        print("ENERGY :",Ef+Ec+El)
        return Ef+Ec+El


    def leftMatrix(V,Wf,Wc,Wl,weight):
        
        AWf = np.zeros(np.shape(np.matmul(np.transpose(Wf[0].toarray()),Wf[0].toarray())))
        for i in range(len(Wf)):
            AWf+=np.matmul(np.transpose(Wf[i].toarray()),Wf[i].toarray())*weight[i]


        AWc = np.zeros(np.shape(np.matmul(np.transpose(Wc[0].toarray()),Wf[0].toarray())))
        for i in range(len(Wc)):
            AWc+=np.matmul(np.transpose(Wc[i].toarray()),Wc[i].toarray())

        AWl = np.zeros(np.shape(np.matmul(np.transpose(Wl[0].toarray()),Wl[0].toarray())))
        for i in range(len(Wl)):
            AWl+=np.matmul(np.transpose(Wl[i].toarray()),Wl[i].toarray())


        return lambda1*AWf,lambda3*AWc,lambda2*AWl


    
    def rightMatrix(V,featuresOrigin,Wf,weight,AWl):
        error = np.zeros(np.shape(np.matmul(np.transpose(Wf[0].toarray()),featuresOrigin[0])))
        for i in range(len(Wf)):
            error+=np.matmul(np.transpose(Wf[i].toarray()),featuresOrigin[i])*weight[i] * lambda1

        right = np.matmul(np.transpose(AWl),error)
        return right

    
    energy_total(V)
    AWf,AWc,AWl = leftMatrix(V,Wf,Wc,Wl,adaptiveFeatureWeight)
    print("LEFT",np.shape(AWf))
    left = np.matmul(np.transpose(AWc),AWc) + np.matmul(np.transpose(AWf),AWf) +np.matmul(np.transpose(AWl),AWl)
    right = rightMatrix(V,featuresOrigin,Wf,adaptiveFeatureWeight,AWl)
    print("RIGHT",np.shape(right))
    res = np.linalg.tensorsolve(left,right)
    print("RES",np.shape(res))
    # visuGrid(res,targetImage)
    print(energy_total(res))
    
        

    

    

### ====================================================================================
### VISU ET RECONSTRUCTION :
### ====================================================================================

def visuGrid(newGrid,image,widthStep=globalWidthStep,heightStep=globalHeightStep):
    V = create_grid(image)
    ymax = np.max(newGrid[0::2])
    xmax = np.max(newGrid[1::2])
    print(xmax,ymax)

    # for i in range(1) :
    grid = newGrid
    
    xmap = grid[1::2]
    ymap = grid[0::2]
    print(xmap)
    print(ymap)
    yminLoc = np.min(ymap)
    xminLoc = np.min(xmap)
    ymaxLoc = np.max(ymap)
    xmaxLoc = np.max(xmap)
    # print(V[0:size:2],ymap)
    # print(V[1:size:2],xmap)
    
    print("?",yminLoc,xminLoc,ymaxLoc,xmaxLoc)


    for i in range(heightStep+1):
        plt.scatter(xmap[i*(widthStep+1):(i+1)*(widthStep+1)],ymap[i*(widthStep+1):(i+1)*(widthStep+1)],s=3)
    plt.show()

    for i in range(widthStep+1):
        plt.scatter(xmap[i::widthStep+1],ymap[i::widthStep+1],s=3)
    plt.show()

    return False

### ====================================================================================
### MAIN
### ====================================================================================

if __name__ == "__main__":
      
    inputPath = "zhang-01/shang"
    # if not os.path.exists(os.path.join(inputPath,"output")):
        # os.makedirs(os.path.join(inputPath,"output"))
    imagesPath = glob.glob(os.path.join(inputPath,"*.jpg"))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.jpeg")))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.png")))
    print(imagesPath)
    print "Opening images"
    images = []
    for compteur in tqdm.tqdm(range(len(imagesPath))) :
        images.append(np.array(cv2.imread(imagesPath[compteur])))

    dicImages = refineGlobal(images,show=False)
    featuresOrigin,featuresTarget = dico_to_array(dicImages[0,1])
    # for featureIndex in range(0,len(featuresOrigin),10):
    #     features = np.transpose(featuresOrigin[featureIndex:featureIndex+10])
    #     plt.scatter(features[0],features[1])
    #     plt.imshow(images[0])
    #     plt.show()
    #     features = np.transpose(featuresTarget[featureIndex:featureIndex+10])
    #     plt.scatter(features[0],features[1])
    #     plt.imshow(images[1])
    #     plt.show()
    solve2(images[1],featuresOrigin,featuresTarget)



        
        
