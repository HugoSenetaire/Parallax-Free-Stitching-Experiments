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
from skimage.measure import label
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import math
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.graph import route_through_array

numSegment = 100
globalWidthStep = 15
globalHeightStep = 15
epsilon = 1e-2
sigma_m = 10
lambdaBig = 1.5
lambdaSmall = 0.1
minDistThreshold = 20
sampleSegmentNumber = 10

lambda1 = 5#5
lambda2 = 1#1
lambda3 = 10 #10



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
            
            M, mask = cv2.findHomography(np.array(dic[label][0]),np.array(dic[label][1]), cv2.RANSAC,5.0)
            if len(np.where(mask.reshape(-1))[0])<=2:
                del dic[label]
            else :
                dic[label][0] = np.array(dic[label][0])[np.where(mask.reshape(-1))]
                dic[label][1] = np.array(dic[label][1])[np.where(mask.reshape(-1))]
            
        
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
            V.append(y*(float(height-1)/heightStep))
            V.append(x*(float(width-1)/widthStep))

    return np.array(V)



### FEATURE TERM :
def adaptive_feature_weight(distanceSeam,distanceReference):
    """ For eWch feature, create the weight associated, 
    NOT TESTED YET"""
    wFeature = []
    for i in range(len(distanceSeam)):
        if distanceSeam[i]<=minDistThreshold:
            wFeature.append(lambdaBig*(np.exp(-(distanceReference[i]**2)/(2*sigma_m**2))+epsilon))
        else :
            wFeature.append(lambdaSmall*(np.exp(-(distanceReference[i]**2)/(2*sigma_m**2))+epsilon))

    return np.array(wFeature)


def find_corners(targetImage,point,heightStep=globalHeightStep,widthStep=globalWidthStep):
    """ For eWch point, find the indices of the four corners associated. 
    NOT TESTED : NEED TO CHECK IF point IS (X,Y) or (Y,X)
    NEED TO CHECK THE RESULT """

    height,width = np.shape(targetImage)[:2]
 
 
    yStep = float(height-1)/heightStep
    xStep = float(width-1)/widthStep

    yIndex = int(np.floor((point[1])/yStep))
    xIndex = int(np.floor((point[0])/xStep))

    
    index = 2*(yIndex*(widthStep+1)+xIndex)

    

    return [index,index+2,index+2*(widthStep+1),index+2*(widthStep+1)+2]

def fill_interpolation_matrice(corner,V,point,heightStep = globalHeightStep,widthStep=globalWidthStep):
    """ For eWch point, fill the matrix indices. 
    point IS X,Y
    SHOULD BE X,Y for point and Y,X for V[CORNER]V[CORNER+1]
    NEED TO CHECK THE RESULT """
    
    
    W = np.zeros((2,len(V)))
    wtot = 1
    
    
    if (point[1]>=V[-2] and point[0]>=V[-1]) \
        or (point[1]>=V[-2] and point[0]<=0)\
        or  (point[1]<=0 and point[0]>=V[-1]) :
        W[0,corner[0]]=1
        W[1,corner[0]+1]=1
        
    elif point[1]>=V[2*(widthStep+1)*(heightStep+1)-2] and V[corner[0]]==V[2*(widthStep+1)*(heightStep+1)-2]:
        w3 = abs(V[corner[1]+1]-point[0])
        w4 = abs(V[corner[0]+1]-point[0])
        wtot = w3+w4
        W[0,corner[1]] = w4
        W[1,corner[1]+1] = w4
        W[0,corner[0]] = w3
        W[1,corner[0]+1] = w3

    elif point[0]>=V[2*(widthStep+1)*(heightStep+1)-1] and V[corner[0]+1]==V[2*(widthStep+1)*(heightStep+1)-1]:
        w2 = abs(V[corner[2]]-point[1])
        w4 = abs(V[corner[0]]-point[1])
        wtot = w2+w4

        W[0,corner[2]] = w4
        W[1,corner[2]+1] = w4
        W[0,corner[0]] = w2
        W[1,corner[0]+1] = w2

    else :
        
        w4 = (V[corner[0]]-point[1])*(V[corner[0]+1]-point[0])
        w3 = -(V[corner[1]]-point[1])*(V[corner[1]+1]-point[0])
        w1 = (V[corner[3]]-point[1])*(V[corner[3]+1]-point[0])
        w2 = -(V[corner[2]]-point[1])*(V[corner[2]+1]-point[0])
        
        if len(np.where(np.array([w1,w2,w3,w4])==0)[0])==2:
            if w1 ==0 and w2 ==0 : 
                w1 = abs(V[corner[3]+1]-point[0])
                w2 = abs(V[corner[2]+1]-point[0])
                wtot = w1+w2

                W[0,corner[3]] = w2
                W[1,corner[3]+1] = w2

                W[0,corner[2]] = w1
                W[1,corner[2]+1] = w1
                

            elif w1 == 0 and w3 ==0 :
                w1 = abs(V[corner[3]]-point[1])
                w3 = abs(V[corner[1]]-point[1])
                wtot = w1+w3

                W[0,corner[3]] = w3
                W[1,corner[3]+1] = w3

                W[0,corner[1]] = w1
                W[1,corner[1]+1] = w1
                
            
            elif w2 == 0 and w4 ==0 :
                w2 = abs(V[corner[2]]-point[1])
                w4 = abs(V[corner[0]]-point[1])
                wtot = w2+w4

                W[0,corner[2]] = w4
                W[1,corner[2]+1] = w4

                W[0,corner[0]] = w2
                W[1,corner[0]+1] = w2
                

            elif w3 == 0 and w4 ==0 :
                w3 = abs(V[corner[1]+1]-point[0])
                w4 = abs(V[corner[0]+1]-point[0])
                wtot = w3+w4
                W[0,corner[1]] = w4
                W[1,corner[1]+1] = w4

                W[0,corner[0]] = w3
                W[1,corner[0]+1] = w3
                
        
        # elif len(np.where([w1,w2,w3,w4]==0)[0])>2 or len(np.where([w1,w2,w3,w4]==0)[0])==1 :
        #     print(len(np.where(np.array([w1,w2,w3,w4])==0)))
        #     print(w1,w2,w3,w4)
        #     raise Exception
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


def create_interpolation_feature_matrices(targetImage,featuresTarget,V,widthStep=globalWidthStep,heightStep=globalHeightStep):
    """ find the interpolation matrix for every features

    """
    W_feature = []

    for point in featuresTarget :
            
            corners = find_corners(targetImage,point,widthStep=widthStep,heightStep=heightStep)
            
            assert(V[corners[0]]<=point[1]and V[corners[0]+1]<=point[0])
            assert(V[corners[1]]<=point[1]and V[corners[1]+1]>=point[0])
            assert(V[corners[2]]>=point[1]and V[corners[2]+1]<=point[0])
            assert(V[corners[3]]>=point[1]and V[corners[3]+1]>=point[0])
            W_feature.append(fill_interpolation_matrice(corners,V,point,widthStep=widthStep,heightStep=heightStep))
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
    # print("UV",UV)
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
    # print("VECTORU,VECTORV",vectorU,vectorV)

    u = np.dot(vectorU,np.matmul(W-Wb,V))
    v = np.dot(vectorV,np.matmul(W-Wb,V))
    # print("UV",[u,v])
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
    # for y in range(1,heightStep+1):
    #     for x in range(1,widthStep+1):
    #         index = 2*(y*(widthStep+1)+x)
    #         indexB = 2*((y-1)*(widthStep+1)+x)
    #         indexC = 2*(y*(widthStep+1)+(x-1))
    #         # index = 2*i
    #         W = np.zeros((2,len(V)))
    #         W[0,index] = 1 #ya sortie y
    #         W[1,index+1] = 1 #xa sortie x

    #         W[0,indexB] =  -1+u#yb sortie y
    #         W[0,indexB+1] = -v #xb sortie y
    #         W[1,indexB] =  v #yb sortie x
    #         W[1,indexB+1] = -1+u   #xb sortie x

    #         W[0,indexC] = -u #yc sortie y
    #         W[0,indexC+1] = +v #xc sortie y
    #         W[1,indexC] = -v #yc sortie x
    #         W[1,indexC+1] = -u #xc sortie x

    #         if init and np.linalg.norm(np.matmul(W,V))>1e-3 :
    #             print("MATRICE W",csr_matrix(W))
    #             print("WHERE",np.where(W!=0))
    #             print("SORTIE",np.matmul(W,V))
    #             raise Exception
            
    #         Wtot.append(csr_matrix(W))

    y = heightStep
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

    x = widthStep
    for y in range(1,heightStep+1):
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
    """
    W_feature = []
    
    for pointAux in points:
            print(pointAux)
            point = np.array(pointAux).reshape(2)
            corners = find_corners(targetImage,point,heightStep=heightStep,widthStep=widthStep)
            aux = copy.deepcopy(fill_interpolation_matrice(corners,V,point,heightStep=heightStep,widthStep=widthStep))
            assert(np.shape(aux)==(2,len(V)))
            assert(aux.dot(V)[0]-point[1]<1e-3 and aux.dot(V)[1]-point[0]<1e-3)
            W_feature.append(aux)
    
    return W_feature



def eliminate_branching_node(contours):
    from collections import defaultdict

    def list_duplicates(seq):
        dico = {}
        for i in range(len(seq)):
            try :
                dico[(seq[i][0],seq[i][1])].append(i)
            except(KeyError):
                dico[(seq[i][0],seq[i][1])] = [i]

        for key in dico.keys() :
            if len(dico[key])==1:
                del dico[key]

        return dico
        # return ((key,locs) for key,locs in dico.items() 
        #                         if len(locs)>1)


    # for dup in sorted(list_duplicates(source)):
    #     print(dup)


    contourAux = []
    branch = True
    while branch :
        branch = False
        print(len(contours))
        for contour in contours :
            reshapedContour  = contour.reshape(-1,2)
            dic_duplicates = list_duplicates(reshapedContour)
            if len(dic_duplicates.keys())>=2:
                branch = True
            if len(dic_duplicates.keys())==0:
                continue
            # print(dic_duplicates.keys())
            # index = np.random.randint(0,len(dic_duplicates.keys()))
            index = 0
            value = 100000
            for k in range(len(dic_duplicates.keys())):
                auxKey = dic_duplicates.keys()[k]
                if abs(dic_duplicates[auxKey][1]-len(contour)/2)<value :
                    index = k
                    value = abs(dic_duplicates[auxKey][1]-len(contour)/2)

    
            keyInit = dic_duplicates.keys()[index]
            
            
            aux1 = reshapedContour[0:dic_duplicates[keyInit][1]+1]
            aux2 = reshapedContour[dic_duplicates[keyInit][1]:]
            if len(aux1)>30 :
                contourAux.append(aux1)
            if len(aux2)>30: 
                contourAux.append(aux2)
        contours = copy.deepcopy(contourAux)
        print(len(contourAux))
        contourAux = []
    return contours

        





def create_curve_points(targetImage,V,showCurve = False,widthStep = globalWidthStep,heightStep=globalHeightStep,test = False) :
    """ CONTOUR CHECKED,
    MATRICES NOT CHECKED"""
    img = targetImage
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    edges = cv2.Canny(img,100,200)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    contours = filter(lambda m: np.shape(m)[0]>50,contours)
    

    if showCurve :
        for contour in contours:
            # print(np.shape(contour))
            # print(contour)
            print("MAX 0 ",np.max(contour.reshape(-1,2)[:,0]))
            print("MAX 1",np.max(contour.reshape(-1,2)[:,1]))
        for i in range(len(contours)):
            aux = cv2.drawContours(img, contours, i, (0,255,0), 3)
            plt.imshow(aux)
            plt.show()


    contours = eliminate_branching_node(contours)
    contours = filter(lambda m: np.shape(m)[0]>30,contours)

    contourMatrix = []
    for contour in contours :
        contourMatrix.append(interpolation_curve_matrices(targetImage,contour[0::sampleSegmentNumber,:],V,widthStep=widthStep,heightStep=heightStep))
        
    if test :
        for contourIndex in range(len(contours)) :
            contour =contours[contourIndex]
            contourAux = contour[0::sampleSegmentNumber,:,:]
            for pointIndex in range(len(contourAux)):
                point = contourAux[pointIndex].reshape(2)
                result = contourMatrix[contourIndex][pointIndex].dot(V)
                assert(abs(point[0]-result[1])<1e-3)
                assert(abs(point[1]-result[0])<1e-3)

    print("NB contours",len(contourMatrix))
    return contourMatrix

def create_straight_line_points(targetImage,V,showCurve = False,widthStep = globalWidthStep,heightStep=globalHeightStep,test = False) :
    img = targetImage
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print("Lines",len(lines))
    contours = []
    for line in lines :
        for x1,y1,x2,y2 in line:
            # xnew = np.arange(x1,x2,10)
            # ynew = np.arange(y1,y2,10)
            xStep = (x2-x1)/11
            yStep = (y2-y1)/11
            contour = []
            for i in range(10):
                contour.append([x1+xStep*i,y1+yStep*i])
            contour.append([x2,y2])
            print(np.shape(contour))
            contours.append(copy.deepcopy(contour))

    # if showCurve :
    #     # for contour in contours:
    #     #     print("MAX 0 ",np.max(contour.reshape(-1,2)[:,0]))
    #     #     print("MAX 1",np.max(contour.reshape(-1,2)[:,1]))
    #     for x1,y1,x2,y2 in lines[0]:
    #         plt.scatter(np.arange(x1,x2,10),np.arange(y1,y2),c="red")
    #     plt.imshow(targetImage)
    #     plt.show()

    contourMatrix = []
    for contour in contours :
        contourMatrix.append(interpolation_curve_matrices(targetImage,contour,V,widthStep=widthStep,heightStep=heightStep))
        
    return contourMatrix


def create_curve_matrices(targetImage,V,showCurve = False,init=False,widthStep=globalWidthStep,heightStep=globalHeightStep,test=False,straightLines = False,bypass=False,contourMatrix = None):
    """ Create Matrix associated with contours optimisation from contour points
    NOT TESTED  """

    if not bypass :
        if straightLines :
            contourMatrix = create_straight_line_points(targetImage,V,showCurve=showCurve,widthStep=widthStep,heightStep=heightStep)
        else :
            contourMatrix = create_curve_points(targetImage,V,showCurve=showCurve,widthStep=widthStep,heightStep=heightStep)

    
    # print(contourMatrix)
    Wtot = []
    if test:
        Wtest_key = []
        Wtest_aux = []
    for contour in contourMatrix:
        

        vectorU = (contour[-1]-contour[0]).dot(V)
        
        vectorV = [vectorU[1],-vectorU[0]]
        
        vectorU = vectorU/(np.linalg.norm(vectorU)**2)
        vectorV = vectorV/(np.linalg.norm(vectorV)**2)

        Wb = contour[0].toarray()
        Wc = contour[-1].toarray()
     

        for i in range(1,len(contour)-1):
            W = np.zeros((2,len(V)))
            Waux = np.zeros((2,len(V)))
            vectorVkey = (contour[i]-contour[0]).dot(V)
            Wkey = contour[i].toarray()
            

            u = np.dot(vectorU,vectorVkey)
            v = np.dot(vectorV,vectorVkey)
            Wdiff = Wc-Wb
            # print(Wdiff)
            W[0] = Wkey[0]-(Wb[0]+u*(Wc[0]-Wb[0])+v*(Wc[1]-Wb[1]))
            W[1] = Wkey[1]-(Wb[1]+u*(Wc[1]-Wb[1])-v*(Wc[0]-Wb[0]))

            if test :
                Waux[0] = (Wb[0]+u*(Wc[0]-Wb[0])+v*(Wc[1]-Wb[1]))
                Waux[1] = (Wb[1]+u*(Wc[1]-Wb[1])-v*(Wc[0]-Wb[0]))
                


            if init and np.linalg.norm(np.matmul(W,V))>1e-3 :
                print("MATRICE W",csr_matrix(W))
                print("WHERE",np.where(W!=0))
                print("VALUE",W[np.where(W!=0)])
                print("SORTIE",np.matmul(W,V))
                raise Exception

            if test :
                Wtest_key.append(csr_matrix(Wkey))
                Wtest_aux.append(csr_matrix(Waux))

            Wtot.append(csr_matrix(W))
    if test :
        return Wtot,Wtest_key,Wtest_aux
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
        # print(i,np.linalg.norm(W[i].dot(V))**2)
        # print(W[i].dot(V))
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

### Essayer avec le seam de l'article !!!

def warpImage(sourceImage,targetImage,gridOptimised,featuresOrigin,featuresTarget,show=False):
    ## Meilleure facon de faire, deja rajouter les pads aux images meme si trop grand ?


    imageA = copy.deepcopy(sourceImage)
    imageB = copy.deepcopy(targetImage)
    # featuresOriginAux = np.array(copy.deepcopy(featuresOrigin))
    # featuresTargetAux = np.array(copy.deepcopy(featuresTarget))

    # Pensez a warpez les features egalement,
    newGrid = copy.deepcopy(gridOptimised)
    xmap = newGrid[1::2]
    ymap = newGrid[0::2]
    yminLoc = np.min(ymap)
    xminLoc = np.min(xmap)
    ymaxLoc = np.max(ymap)
    xmaxLoc = np.max(xmap)
    print(yminLoc,xminLoc,ymaxLoc,xmaxLoc)

    if xminLoc <0 :
        left = False # Source image is on the right :
        newGrid[1::2] -= np.ceil(xminLoc)
        # Pad source Image on the left :
        print(np.ceil(xminLoc))
        # imageA = np.concatenate((np.zeros((np.shape(imageA)[0],int(np.ceil(-xminLoc)),3),dtype=float),np.ones(np.shape(imageA))*128),axis=1)
        imageA = np.pad(imageA,((0,0),(int(np.ceil(-xminLoc)),0),(0,0)),mode = "constant")
        # featuresOriginAux[:,1]+=ceil(xminLoc)
     

    else :
        left = True # Source image is on the left
        #Pad source Image on the right :
        imageA = np.concatenate(imageA,(np.zeros(np.shape(imageA))),axis=1)

    
    if yminLoc<0 : # Source image is on the bottom
        newGrid[0::2] -=np.ceil(yminLoc)
        # Pad Source image on the top
        # imageA = np.concatenate((np.zeros((int(np.ceil(-yminLoc)),np.shape(imageA)[1],3),dtype=float),imageA),axis=0)
        imageA = np.pad(imageA,((int(np.ceil(-yminLoc)),0),(0,0),(0,0)),mode = "constant")
        
        # featuresOriginAux[:,0]+=ceil(yminLoc)

    
    else :   # Source image is on the top
        # Pad Source image on the bottom
        imageA = np.concatenate(imageA,(np.zeros(np.shape(imageA))),axis=0,dtype=np.uint8)

    V = np.array(create_grid(imageB))
    V = V.reshape(-1,2)
    V = np.transpose(V)
    V = np.transpose([V[1],V[0]])

    grid = newGrid.reshape(-1,2)
    grid = np.transpose(grid)
    grid = np.transpose([grid[1],grid[0]])
    
    rows, cols = imageA.shape[0], imageA.shape[1]

    tform = PiecewiseAffineTransform()
    tform.estimate(V, grid)
    imageBwarped = warp(imageB, tform,output_shape = (rows,cols))
    
    imageA = np.array(imageA,dtype=np.uint8)
    imageBwarped = np.array(np.array(imageBwarped)*256,dtype =np.uint8)

    return imageA,imageBwarped,left,newGrid


    


def seam_2_images(warpedA, warpedB, width_factor=1):
    """ Seams two images into one
    Parameters
    ==========
    warpedA: (nparray) - The image on the left
    warpedB: (nparray) - The image on the right
    width_factor: (int) - The width used to compute the seaming mask [optional]
    Returns
    =======
    combined: (nparray) - The seamed image
    """

    def find_seam_mask(warpedA, warpedB, grayA, grayB, width_factor=1):

        def generate_costs(grayA, grayB, overlap):
            xmin = np.min(overlap[0])
            xmax = np.max(overlap[0])
            ymin = np.min(overlap[1])
            ymax = np.max(overlap[1])
            costs = np.ones((xmax - xmin + 1, ymax - ymin + 1))
            costs[(overlap[0] - xmin, overlap[1] - ymin)] = np.abs(grayA[(overlap[0], overlap[1])] - grayB[(overlap[0], overlap[1])]) / (3 * 255.0)
            costs[0, :] = 0
            costs[-1, :] = 0
            return costs

        def shortest_path(costs):
            pts, _ = route_through_array(costs, (0, costs.shape[1] // 2), (costs.shape[0] - 1, costs.shape[1] // 2), fully_connected=False, geometric=False)
            pts = np.array(pts)
            return pts

        # Resized Gray
        # grayA = imutils.resize(grayA, width=int(warpedA.shape[1] * width_factor))
        # grayB = imutils.resize(grayB, width=int(warpedB.shape[1] * width_factor))

        # Find path
        overlap = np.where((grayA > 0) * (grayB > 0))
        if overlap[0].size == 0:
            return np.zeros((warpedA.shape[0], warpedA.shape[1]))
        costs = generate_costs(grayA, grayB, overlap)
        path = shortest_path(costs)

        # Overlap Seam mask
        overlap_seam_mask = np.zeros((costs.shape[0], costs.shape[1] + 1))
        overlap_seam_mask[path[:, 0], path[:, 1] + 1] = 1.0
        overlap_seam_mask = label(overlap_seam_mask, connectivity=1, background=-1)

        # Complete Seam mask
        xmin = np.min(overlap[0])
        xmax = np.max(overlap[0])
        ymin = np.min(overlap[1])
        ymax = np.max(overlap[1])

   

        if (grayA.shape[1] - ymax <= 1):
            return np.zeros((warpedA.shape[0], warpedA.shape[1])) + 1

        seam_mask = np.zeros(grayA.shape)
        seam_mask[xmin:xmax + 1, ymin:ymax + 2] = overlap_seam_mask
        seam_mask = cv2.resize(seam_mask, (warpedA.shape[1], warpedA.shape[0]), interpolation=cv2.INTER_NEAREST)

        return seam_mask

    # final_mask = np.zeros(warpedA.shape[0], warpedA.shape[1])
    # Gray
    grayA = cv2.cvtColor(warpedA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(warpedB, cv2.COLOR_BGR2GRAY)



    # Init Combine
    combined = warpedA
    final_mask = (grayA > 0).astype(np.uint8)
    combined[(grayA == 0) * (grayB > 0)] = warpedB[(grayA == 0) * (grayB > 0)]
    final_mask[(grayA == 0) * (grayB > 0)] = 3

    # Compute mask
    seam_mask = find_seam_mask(warpedA, warpedB, grayA, grayB, width_factor=width_factor)

    # Apply mask
    combined[(grayB > 0) * (seam_mask >= 3)] = warpedB[(grayB > 0) * (seam_mask >= 3)]

    final_mask[(grayB > 0) * (seam_mask >= 3)] = 3


    # Turn into np.uint8 (the mask is already in np.uint8)
    combined = combined.astype(np.uint8)

    return combined,final_mask



def get_distance_seam(sourceImage,targetImage,gridOptimised,featuresOrigin,featuresTarget,wFeature,method = "Yoobic",show = False):
    imageA,imageB,left,newGrid = warpImage(sourceImage,targetImage,gridOptimised,featuresOrigin,featuresTarget)


    if method == "Yoobic":
        if left :
            combine, final_mask = seam_2_images(imageA, imageB)
        else :
            combine, final_mask = seam_2_images(imageB,imageA)
    else :
        print("Not implemented")
        raise Exception
    if show :
        plt.imshow(combine)
        plt.show()
        plt.imshow(final_mask)
        plt.show()

    grad = np.linalg.norm(np.gradient(final_mask),axis =0)*100
    if show:
        plt.imshow(grad)
        plt.show()
        
        print(np.shape(grad))
    pointsSeam = np.transpose(np.where(grad!=0))
    print(pointsSeam)


    dist = []
    for featureMatrix in wFeature :
        feature = featureMatrix.dot(newGrid)
        feature = np.array([feature[1],feature[0]])
        distance = np.min(np.linalg.norm(np.subtract(pointsSeam,feature),axis=1))
        
        dist.append(distance)
        # distance = grad = 
    
    
    
    return combine,final_mask,dist


### ====================================================================================
### SOLVING
### ====================================================================================


### SOLVE FOR ONE ITERATION:

def solve(sourceImage,targetImage,featuresOrigin,featuresTarget):
    V =create_grid(targetImage)

    Wf = create_interpolation_feature_matrices(targetImage,featuresTarget,V)
    Wc1 = create_curve_matrices(targetImage,V,init=True,straightLines=True,showCurve=True)
    Wc2 = create_curve_matrices(targetImage,V,init=True,straightLines=False,showCurve=False)
    Wc = np.concatenate((Wc1,Wc2),axis =0)
    Wl = create_local_preserving_matrices(targetImage,V,init=True)
    distanceSeam = np.zeros((len(featuresOrigin)))
    features = []
    for i in range(len(Wf)):
        aux = Wf[i].dot(V)
        features.append([aux[1],aux[0]])

    features = np.array(features)
    distanceReference = np.linalg.norm(featuresOrigin-features,axis=1)
    adaptiveFeatureWeight=copy.deepcopy(adaptive_feature_weight(distanceSeam,distanceReference))
    # adaptiveFeatureWeight=np.ones(np.shape(adaptiveFeatureWeight))
    assert(len(featuresOrigin)==len(featuresTarget))
    
  
    def energy_total(V): 
        Ef = energyFeature(Wf,V,featuresOrigin,adaptiveFeatureWeight)
        # print("ENERGY FEATURE :",Ef)
        Ec = energyCurve(Wc,V)
        # print("ENERGY CURVE :",Ec)
        El = energyLocal(Wl,V)
        # print("ENERGY LOCAL :",El)
        # print("ENERGY :",Ef+Ec+El)
        return Ef+Ec+El


    
    test = np.mean(featuresOrigin-featuresTarget,axis=0)
    # print(test)
    # print(energy_total(V))
    # for i in range(len(V)):
        # print(V[i],test[(i+1)%2])
        # V[i] = V[i]+test[(i+1)%2]
    # visuGrid(V,targetImage,heightStep=globalHeightStep,widthStep=globalWidthStep,showFeature=True,Wf=Wf,featuresOrigin=featuresOrigin)
    # print(energy_total(V))

    def grad(V):
        auxF = []
        
        for i in range(len(featuresOrigin)):
            feature = [featuresOrigin[i][1],featuresOrigin[i][0]]
            auxF.append(Wf[i].dot(V)-feature)

        gradF = np.zeros(np.shape(V))
        for i in range(len(featuresOrigin)):
            # print(np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i])
            gradF +=np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i]
        # print("GRADF",np.linalg.norm(lambda1*gradF))

        # print("GradF",gradF)
        gradC = np.zeros(np.shape(V))
        for i in range(len(Wc)):
            # print(np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V)))
            gradC+=np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V))
        # print("GRADC",np.linalg.norm(lambda3*gradC))
        # print("GradC",gradC)
        gradL= np.zeros(np.shape(V))
        for i in range(len(Wl)):
            gradL+=np.matmul(np.transpose(Wl[i].toarray()),Wl[i].dot(V))
        # print("GRADL",np.linalg.norm(lambda2*gradL))
        # print("gradL",gradL)
         

        

        return (lambda1*gradF+lambda3*gradC+lambda2*gradL)
    # grad(V)
    # res,f,info = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy_total,V,approx_grad=True,iprint = 99 ))
    # reconstructionV2(res,targetImage)
    # visuGrid(res,targetImage,heightStep=globalHeightStep,widthStep=globalWidthStep,showFeature=True,Wf=Wf,featuresOrigin=featuresOrigin)
    # energy_total(res)
    

  




    res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy_total,V,fprime=grad,iprint = 99,maxiter= 200 ))
    combine,mask,distanceSeam = get_distance_seam(sourceImage,targetImage,res[0],featuresOrigin,featuresTarget,Wf)

    for i in range(2):
        V = res[0]
        adaptiveFeatureWeight=copy.deepcopy(adaptive_feature_weight(distanceSeam,distanceReference))
        res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy_total,V,fprime=grad,iprint = 99,maxiter= 200 ))
        combine,mask,distanceSeam = get_distance_seam(sourceImage,targetImage,res[0],featuresOrigin,featuresTarget,Wf)

    # reconstructionV2(res[0],targetImage)
    # visuGrid(res[0],targetImage,heightStep=globalHeightStep,widthStep=globalWidthStep,showFeature=True,Wf=Wf,featuresOrigin=featuresOrigin)
    # energy_total(res[0])
    return combine


def solve2(targetImage,featuresOrigin,featuresTarget):
    V =create_grid(targetImage)
    
    distanceReference = np.linalg.norm(featuresOrigin-featuresTarget,axis=1)
    print(distanceReference)
    distanceSeam = np.zeros(np.shape(distanceReference))

    adaptiveFeatureWeight=adaptive_feature_weight(distanceSeam,distanceReference)
    # adaptiveFeatureWeight=np.ones(np.shape(adaptiveFeatureWeight))
    # print("WEIGHT",adaptiveFeatureWeight)
    Wf = create_interpolation_feature_matrices(targetImage,featuresTarget,V)
    Wc1 = create_curve_matrices(targetImage,V,init=True,straightLines=True,showCurve=True)
    Wc2 = create_curve_matrices(targetImage,V,init=True,straightLines=False,showCurve=False)
    Wc = np.concatenate((Wc1,Wc2),axis =0)
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

        # WfAux = np.zeros(np.shape(Wf[0]))
        # for i in range(len(Wf)):
        #     WfAux += Wf[i].toarray()*weight[i]
        # print(np.shape(WfAux))
        # AWfAux = np.matmul(np.transpose(WfAux),WfAux)
        
        
        # WcAux = np.zeros(np.shape(Wc[0]))
        # for i in range(len(Wc)):
        #     WcAux = Wc[i].toarray()
        
        # AWcAux = np.matmul(np.transpose(WcAux),WcAux)

        # WlAux = np.zeros(np.shape(Wl[0]))
        # for i in range(len(Wl)):
        #     WlAux += Wl[i].toarray()
        
        # AWlAux = np.matmul(np.transpose(WlAux),WlAux)
        
        AWc = np.zeros(np.shape(np.matmul(np.transpose(Wc[0].toarray()),Wf[0].toarray())))
        for i in range(len(Wc)):
            AWc+=np.matmul(np.transpose(Wc[i].toarray()),Wc[i].toarray())


        AWl = np.zeros(np.shape(np.matmul(np.transpose(Wl[0].toarray()),Wl[0].toarray())))
        for i in range(len(Wl)):
            AWl+=np.matmul(np.transpose(Wl[i].toarray()),Wl[i].toarray())


        return (lambda1)*AWf,(lambda3)*AWc,(lambda2)*AWl


    
    def rightMatrix(V,featuresOrigin,Wf,weight,AWl):
        error = np.zeros(np.shape(np.matmul(np.transpose(Wf[0].toarray()),featuresOrigin[0])))
        for i in range(len(Wf)):
            feature = np.array([featuresOrigin[i][1],featuresOrigin[i][0]])
            error+=np.matmul(np.transpose(Wf[i].toarray()),featuresOrigin[i])*weight[i] 
        # right = np.matmul(np.transpose(AWl),error) * lambda1
        right = error * lambda1
        return right

    def grad(V):
            auxF = []
            
            for i in range(len(featuresOrigin)):
                feature = [featuresOrigin[i][1],featuresOrigin[i][0]]
                auxF.append(Wf[i].dot(V)-feature)

            gradF = np.zeros(np.shape(V))
            for i in range(len(featuresOrigin)):
                # print(np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i])
                gradF +=np.matmul(np.transpose(Wf[i].toarray()),auxF[i])*adaptiveFeatureWeight[i]
            print("GRADF",np.linalg.norm(lambda1*gradF))

            # print("GradF",gradF)
            gradC = np.zeros(np.shape(V))
            for i in range(len(Wc)):
                # print(np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V)))
                gradC+=np.matmul(np.transpose(Wc[i].toarray()),Wc[i].dot(V))
            print("GRADC",np.linalg.norm(lambda3*gradC))
            # print("GradC",gradC)
            gradL= np.zeros(np.shape(V))
            for i in range(len(Wl)):
                gradL+=np.matmul(np.transpose(Wl[i].toarray()),Wl[i].dot(V))
            print("GRADL",np.linalg.norm(lambda2*gradL))
    
    print("W0",energy_total(V))
    AWf,AWc,AWl = leftMatrix(V,Wf,Wc,Wl,adaptiveFeatureWeight)
    print("LEFT",np.shape(AWf),np.shape(AWc),np.shape(AWl))
    # left = np.matmul(np.transpose(AWc),AWc) + np.matmul(np.transpose(AWf),AWf) +np.matmul(np.transpose(AWl),AWl)
    left = AWf+AWc+AWl 
    # leftAux = np.zeros((np.shape(AWf)[0]+np.shape(AWc)[0]+np.shape(AWl)[0],np.shape(AWf)[1]))
    # print(np.shape(leftAux))
    # left = np.concatenate((AWf,AWc,AWl),axis =1)
    print(np.shape(left))
    right = rightMatrix(V,featuresOrigin,Wf,adaptiveFeatureWeight,AWl)
        
   
    
    res = np.linalg.tensorsolve(left,right)
    # res,residuals,rank,s = numpy.linalg.lstsq(left,right)
    # res = np.matmul(np.linalg.inverse(left),right)
    print("GRAD")
    grad(V)
    grad(res)
    print("RES",np.shape(res))
    visuGrid(res,targetImage)
    print("E1",energy_total(res))

    print("ECART",np.linalg.norm(V-res))
    
        

    

    

### ====================================================================================
### VISU ET RECONSTRUCTION :
### ====================================================================================

def visuGrid(newGrid,image,widthStep=globalWidthStep,heightStep=globalHeightStep,showFeature = False,Wf = None,featuresOrigin=None):
    V = create_grid(image)
    ymax = np.max(newGrid[0::2])
    xmax = np.max(newGrid[1::2])
    print(xmax,ymax)

    # for i in range(1) :
    grid = newGrid
    
    xmap = grid[1::2]
    ymap = grid[0::2]
    # print(xmap)
    # print(ymap)
    yminLoc = np.min(ymap)
    xminLoc = np.min(xmap)
    ymaxLoc = np.max(ymap)
    xmaxLoc = np.max(xmap)
    # print(V[0:size:2],ymap)
    # print(V[1:size:2],xmap)
    
    # print("?",yminLoc,xminLoc,ymaxLoc,xmaxLoc)


    for i in range(heightStep+1):
        plt.scatter(xmap[i*(widthStep+1):(i+1)*(widthStep+1)],ymap[i*(widthStep+1):(i+1)*(widthStep+1)],s=3)
    plt.show()

    for i in range(widthStep+1):
        plt.scatter(xmap[i::widthStep+1],ymap[i::widthStep+1],s=3)
    plt.show()

    if showFeature :
        for i in range(0,len(featuresOrigin),20):
            pt =Wf[i].dot(newGrid)
            pt2 = [featuresOrigin[i][1],featuresOrigin[i][0]]
            plt.scatter(pt[1],pt[0],s=4,c='red')
            plt.scatter(pt2[1],pt[0],s=4,c='blue')
            plt.scatter(xmap,ymap,s=3,c='black')
            plt.show()


    return False


def reconstructionV2(newGrid,image):
    V = np.array(create_grid(image))

    V = V.reshape(-1,2)
    V = np.transpose(V)
    V = np.transpose([V[1],V[0]])
    
    newImages = []
        
    grid = newGrid.reshape(-1,2)
    grid = np.transpose(grid)
    grid = np.transpose([grid[1],grid[0]])
    
    rows, cols = image.shape[0], image.shape[1] 

    tform = PiecewiseAffineTransform()
    tform.estimate(V, grid)
    # for k in range(len(V)):
        # print(V[k],grid[k])

    out = warp(image, tform,output_shape = (rows,cols))
    
    # fig, ax = plt.subplots()
    # ax.imshow(out)
    # ax.plot(tform.inverse(V)[:, 0], tform.inverse(V)[:, 1], '.b')
    # plt.show()
    newImages.append(out)
    return newImages

### ====================================================================================
### MAIN
### ====================================================================================

if __name__ == "__main__":
    # inputPath = "SimpleWideBaseline2"
    inputPath = "zhang-01/shang"
    # inputPath = "../zhang-08"
    # if not os.path.exists(os.path.join(inputPath,"output")):
        # os.makedirs(os.path.join(inputPath,"output"))
    imagesPath = glob.glob(os.path.join(inputPath,"*.jpg"))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.jpeg")))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.png")))
    imagesPath.sort()
    print(imagesPath)
    print "Opening images"
    images = []
    for compteur in tqdm.tqdm(range(len(imagesPath))) :
        images.append(np.array(cv2.imread(imagesPath[compteur])))
    
    imageAux = copy.deepcopy(images)
    result = imageAux[0]
    for i in range(1,len(imageAux)) :
        images = [result,imageAux[i]]
        dicImages = refineGlobal(images,show=False)
        featuresOrigin,featuresTarget = dico_to_array(dicImages[0,1])
        featuresOrigin,index = np.unique(featuresOrigin,axis =0,return_index = True)
        featuresTarget = featuresTarget[index]
        featuresTarget,index = np.unique(featuresTarget,axis=0,return_index = True)
        featuresOrigin = featuresOrigin[index]
        result = solve(images[0],images[1],featuresOrigin,featuresTarget)
        plt.imshow(result)
        plt.show()



        
        
