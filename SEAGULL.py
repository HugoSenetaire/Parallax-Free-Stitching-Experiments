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

numSegment = 100

# def cvt_to_int(x):
#     if x-floor(x)<0.5:
#         return floor(x)
#     else :
#         return ceil(x)

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


    for indexImage in range(len(images)-1):

        shape = np.shape(images[0])

        # Get Features
        features1 = detect_and_describe(images[indexImage])
        features2 = detect_and_describe(images[indexImage+1])
        ptsA,ptsB,matches = featureMatching(features1,features2,images[indexImage],images[indexImage+1])

        print(len(ptsA),len(ptsB))
        ptsA = ptsA.astype(int)
        ptsB = ptsB.astype(int)

  
        # Get Segmentation :
        # segA = segmentation(images[i])
        segB = segmentation(images[indexImage+1])
       
        # print(np.shape(segB))
        # plt.imshow(np.array(segB)*50)
        # plt.show()
        
        dic = getLabeling(ptsA,ptsB,segB)
        
        for label in dic.keys():
            # print("BEFORE",dic[label][0])
            M, mask = cv2.findHomography(np.array(dic[label][0]),np.array(dic[label][1]), cv2.RANSAC,5.0)
            if len(np.where(mask.reshape(-1))[0])<=2:
                del dic[label]
            else :
                dic[label][0] = np.array(dic[label][0])[np.where(mask.reshape(-1))]
                dic[label][1] = np.array(dic[label][1])[np.where(mask.reshape(-1))]
            # print("AFTER",dic[label][0])
        
        labelValues = list(dic.keys())
        labelValues = sorted(labelValues,key = lambda l : len(dic[l][0]),reverse= True)

        print("DICTIONNAIRE",dic)

        neighboursMap = get_neighbouring_label(segB)
        for key in neighboursMap.keys():
            if key not in labelValues :
                del neighboursMap[key]
            else :
                neighboursMap[key] = np.intersect1d(neighboursMap[key],labelValues)
                    
        # print(labelValues)
        # print(neighboursMap)

        # groupped = [False]*len(labelValues)
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



        #SUPER GROUP :
        groupValues = list(dicGroup.keys())
        groupValues = sorted(groupValues,key = lambda l : len(dicGroup[l][0]),reverse= True)

        dicSuperGroup = {}
        superGroupValue = [-1]*len(groupValues)
        print(len(groupValue))
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
                        if np.all(mask.reshape(-1)) :#or len(np.where(mask.reshape(-1))[0])>len(dicSuperGroup[superGroup][0])+0.5*len(dicGroup[j][0]) :
                            # print("TROUVE",root,currentLabel)
                            groupValue[currentLabelIndex]= group
                            toTreat.extend(list(neighboursMap[currentLabel]))
                            dicSuperGroup[superGroup][0] = np.concatenate((dicSuperGroup[superGroup][0],dicGroup[j][0]))[mask.reshape(-1)]
                            dicSuperGroup[superGroup][1] = np.concatenate((dicSuperGroup[superGroup][1],dicGroup[j][1]))[mask.reshape(-1)]


        print(len(dicSuperGroup.keys()))
        
        # res = cv2.drawMatchesKnn(images[i],features1[2],images[i+1],features2[2],matches,None) 
        # for key in dicSuperGroup.keys():

        #     res = cv2.drawMatchesKnn(images[indexImage],features1[2],images[indexImage+1],features2[2],[],None) 
        #     plt.imshow(res)
        #     for k in range(len(dicSuperGroup[key][0])) :
        #         ptsA = dicSuperGroup[key][0][k]
        #         ptsB = dicSuperGroup[key][1][k]
        #         plt.plot([ptsA[0],ptsB[0]+shape[1]],[ptsA[1],ptsB[1]],linewidth = 1)
        #     plt.show()


        



        
        
