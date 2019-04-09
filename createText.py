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
import graph_tool
import graph_tool.flow

def calculatePotentialCost(images):
    potentialTotal = []
    potential = np.zeros(np.shape(images)[:-1])
    # print(np.shape(potential))
    for i in tqdm.tqdm(range(np.shape(images)[0])):
        potential[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)
        potentialTotal.append(copy.deepcopy(potential))
    return np.array(potentialTotal)

def putNone(image):
    for i in range(len(image)):
        for j in range(len(image)):
            if image[i,j,0] == 0 and image[i,j,1] == 0 and image[i,j,2] == 0 :
                image[i,j] == None

    return image 


def normalize(distance):
    for i in range(len(distance[0])):
        for j in range(len(distance[0][0])):

            Inf = True
            for k in range(len(distance)):
                if not isinf(distance[k][i][j]) or distance[k][i][j]< 1000:
                    Inf = False
                   

            if Inf :
                for k in range(len(distance)):
                    distance[k][i][j] = 0
            else :
                for k in range(len(distance)):
                    if isinf(distance[k][i][j]) or distance[k][i][j]> 1000:
                        distance[k][i][j] = 1000

                
    return distance



def calculateMap(images):

    medianImage = np.zeros(np.shape(images[0]))
    for i in tqdm.tqdm(range(len(medianImage))):
        for j in range(len(medianImage[i])):
            for k in range(3):
                listValue = []
                for imageIndex in range(len(images)):
                    if images[imageIndex][i,j,k]!= None :
                        listValue.append(images[imageIndex][i,j,k])
                medianImage[i,j,k] = np.median(listValue)

    possibleResult = np.abs(np.subtract(images,medianImage))
    # print("POSSIBLE",possibleResult[np.where(possibleResult>1)])

    MAD = np.median(possibleResult,axis = 3)
    # print(MAD[np.where(MAD>3)])
    H = np.where(MAD<10,np.linalg.norm(possibleResult),0)
    # print(H[np.where(H>0)])
    return H
    
    





    





def putEdgeWeight(basicGraph,distance,potential,disparity,labelNumber,maxCols,maxLines):
    nbVertices = maxCols*maxLines+2 

    for i in tqdm.tqdm(range(1,nbVertices-1)):
        currentVertex = basicGraph.vertex(i)
        x,y = getCoords(i,maxCols,maxLines)
        edges = basicGraph.get_in_edges(currentVertex)
        for edge in edges :
            edgeNumber = edge[2]
            vertexFrom = edge[0]
            if vertexFrom ==0 :
                test = distance[disparity[x,y],x,y]
                basicGraph.edge_properties["w"][edge] = distance[disparity[x,y],x,y]
            else :
                xFrom,yFrom = getCoords(vertexFrom,maxCols,maxLines)
                # print(xFrom,yFrom,x,y,)
                # print(disparity[x,y],disparity[xFrom,yFrom])
                # print(labelNumber)
                # print(np.shape(potential))
                basicGraph.edge_properties["w"][edge] = (potential[disparity[xFrom,yFrom],labelNumber,x,y] + potential[disparity[xFrom,yFrom],labelNumber,xFrom,yFrom])


    puit = basicGraph.vertex(nbVertices-1)
    edges = basicGraph.get_in_edges(puit)
    for edge in edges :
        edgeNumber = edge[2]
        vertexFrom = edge[0]
        x,y = getCoords(vertexFrom,maxCols,maxLines)
        basicGraph.edge_properties["w"][edge] = distance[disparity[x,y],x,y]


def createCurrentPotent(potential,disparity,labelNumber):
    potent = np.zeros((np.shape(disparity)[0],np.shape(disparity)[1]))
    for i in range(len(potent)):
        for j in range(len(potent[0])):
            potent[i,j] = potential[disparity[i,j],labelNumber,i,j]
    return potent

def createCurrentDist(dataCost,disp):
    currentDataCost = np.zeros((np.shape(disp)[0],np.shape(disp)[1]))
    for i in range(len(currentDataCost)):
        for j in range(len(currentDataCost[0])):
            currentDataCost[i,j] = dataCost[disp[i,j],i,j]
    return currentDataCost







## MAIN :
if __name__ == "__main__":

    inputPath = "outputVideo1"
    if not os.path.exists(os.path.join(inputPath,"output")):
        os.makedirs(os.path.join(inputPath,"output"))
    imagesPath = glob.glob(os.path.join(inputPath,"*.jpg"))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.jpeg")))
    imagesPath.extend(glob.glob(os.path.join(inputPath,"*.png")))
    print "Opening images"
    images = []
    for compteur in tqdm.tqdm(range(len(imagesPath))) :
        images.append(np.array(cv2.imread(os.path.join(inputPath,"img{}.jpg".format(compteur)))))   
    images = np.array(images)



    
    print "Calculate cost H"
    H = calculateMap(images)
    averageImage = []
    H = np.zeros((np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))

    print "Calculate cost distance"
    with open(os.path.join(inputPath,"pickleAux/distance.pck"),"rb") as f :
        distance = pickle.load(f)

    print("Normalize distance")  
    distance = np.array(normalize(distance))

    #Algorithm
    lambda_value = 100
    gamma_value = 0
    print "Begin minimisation"


    maxCols = np.shape(images)[1]
    maxLines = np.shape(images)[2]
    nbVertices = maxCols*maxLines+2


    # Calculate potential :
    potential = calculatePotentialCost(images)
    disparity = np.random.randint(0,len(images),(np.shape(images)[1],np.shape(images)[2]))

    # Create graph :
    # basicGraph = createGraph(images)
    basicGraph,dic = createGraphFast(images)
    source = basicGraph.vertex(0)
    puit = basicGraph.vertex(nbVertices-1)





    # newImage = reconstructionImage(images,disparity_est)
    
    
    # cv2.imwrite("output3/output/MRFCOST.jpeg",newImage)