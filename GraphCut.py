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
import graph_tool.draw


def compute_energy(lambda_value,distance,gamma_value,H,disparity, potential,details = False):
    # print("compute_energy")
    energy = 0
    energyDistance = 0
    energyMAD = 0
    energyPotential = 0
    energyDataCost = 0
    # print(np.shape(potential))
    for i in range(1,len(disparity)-1):
        for j in range(1,len(disparity[0])-1):
            # if isinf(data_cost[i,j,disparity[i,j]]):
            # print(i,j,disparity[i,j],disparity[i,j-1])
                # continue
            aux1 = lambda_value * distance[disparity[i,j],i,j]
            aux3 = gamma_value * H[disparity[i,j],i,j]
            aux2 =  potential[disparity[i,j],disparity[i+1,j],i,j] + potential[disparity[i+1,j],disparity[i,j],i+1,j] \
                +potential[disparity[i,j],disparity[i-1,j],i,j] + potential[disparity[i-1,j],disparity[i,j],i-1,j]\
                +potential[disparity[i,j],disparity[i,j+1],i,j] + potential[disparity[i,j+1],disparity[i,j],i,j+1] \
                +potential[disparity[i,j],disparity[i,j-1],i,j] + potential[disparity[i,j-1],disparity[i,j],i,j-1]

            # aux2 =  potential[disparity[i,j],disparity[i+1,j],i,j]  \
            #     +potential[disparity[i,j],disparity[i-1,j],i,j] \
            #     +potential[disparity[i,j],disparity[i,j+1],i,j]  \
            #     +potential[disparity[i,j],disparity[i,j-1],i,j] 
            # aux4 = data_cost[i,j,disparity[i,j]]
            # print(aux)
            if isnan(aux1) or isinf(aux1):
                aux1 = 0
            if isnan(aux2) or isinf(aux2) :
                aux2 = 0
            if isnan(aux3) or isinf(aux3):
                aux3 = 0
            # if isnan(aux4) or isinf(aux4):
                # aux4 = 0

            # if aux1<0 or aux2<0 or aux3<0 or aux4<0 :
            # if aux1<0 or aux2<0 or aux3<0 :
                # print("NEGATIF",i,j)
            # if aux<0 : print(data_cost[i,j,disparity[i,j]],potentialUp[disparity[i,j],disparity[i+1,j],i,j] )
            energy+=(aux2 +aux1 + aux3)
            energyDistance +=aux1
            energyMAD+=aux3
            energyPotential+=aux2
            energyDataCost+=(aux1 + aux3)
    
    if details :
        print("Energy Distance", energyDistance)
        print("Energy MAD", energyMAD)
        print("energyPotential",energyPotential)
        print("Energy Data Cost", energyDataCost)

    
    return energy

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
                if not isinf(distance[k][i][j]) or distance[k][i][j]< 100:
                    Inf = False
                   

            if Inf :
                for k in range(len(distance)):
                    distance[k][i][j] = 0
            else :
                for k in range(len(distance)):
                    if isinf(distance[k][i][j]) or distance[k][i][j]> 100:
                        distance[k][i][j] = 100000

                
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
                medianImage[i,j,k] = np.median(copy.deepcopy(listValue))

    possibleResult = np.abs(np.subtract(images,medianImage))
    # print("POSSIBLE",possibleResult[np.where(possibleResult>1)])

    MAD = np.median(possibleResult,axis = 3)
    # print(MAD[np.where(MAD>3)])
    H = np.where(MAD<10,np.linalg.norm(possibleResult),0)
    # print(H[np.where(H>0)])
    return H
    
    



def reconstructionImage(images,disparity):
    new_image = np.zeros(np.shape(images[0]))
    # print(np.shape(images))
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            for k in range(len(new_image[i,j])):
                new_image[i,j,k] = images[disparity[i,j],i,j,k]

    return new_image
            

    

def calculateAverageImage(images):
    new_image = np.zeros(np.shape(images[0]))
    for i in tqdm.tqdm(range(len(new_image))):
        for j in range(len(new_image[i])):
            # for k in range(len(new_image[i,j])):
            nbImage = 0
            average = np.zeros(3)
            for k in range(len(images)):
                if not np.all(images[k,i,j,:] == 0):
                    average+=images[k,i,j,:] 
                    nbImage+=1
                    
                
            if nbImage>0 :
                average=average/float(nbImage)
                # print(average)
            # for k in range(3):



            new_image[i,j] = average
    return new_image

def createGraph(images):
    maxCols = np.shape(images)[1]
    maxLines = np.shape(images)[2]
    nbVertices = maxCols*maxLines+2

    basicGraph = graph_tool.Graph()
    basicGraph.add_vertex(nbVertices)

    source = basicGraph.vertex(0)
    puit = basicGraph.vertex(nbVertices-1)



    for vertexIndex in tqdm.tqdm(range(1,nbVertices-1)):
        currentVertex = basicGraph.vertex(vertexIndex)


        # Source
        basicGraph.add_edge(source,currentVertex)
        # Puit 
        basicGraph.add_edge(currentVertex,puit)

        if (vertexIndex-1)%maxCols != maxCols - 1 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex+1))

        if (vertexIndex-1)%maxCols !=0 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex-1))

        if (vertexIndex-1)/maxCols != maxLines-1 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex+maxCols))
        
        if (vertexIndex-1)/maxCols != 0 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex-maxCols))
    
    propertyGraph = basicGraph.new_edge_property("float")
    basicGraph.edge_properties["w"] = propertyGraph
    return basicGraph

def createGraphFast(images):
    maxCols = np.shape(images)[1]
    maxLines = np.shape(images)[2]
    nbVertices = maxCols*maxLines+2

    basicGraph = graph_tool.Graph()
    basicGraph.add_vertex(nbVertices)

    source = basicGraph.vertex(0)
    puit = basicGraph.vertex(nbVertices-1)
    index =0
    dic = {"source":[],"puit":[],"right":[],"left":[],"down":[],"up":[]}
    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        # Source
        basicGraph.add_edge(source,currentVertex)
        dic["source"].append(index)
        index+=1

    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        # Puit 
        basicGraph.add_edge(currentVertex,puit)
        dic["puit"].append(index)
        index+=1



    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        if (vertexIndex-1)%maxCols != maxCols - 1 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex+1))
            dic["right"].append(index)
            index+=1
    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        if (vertexIndex-1)%maxCols !=0 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex-1))
            dic["left"].append(index)
            index+=1
    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        if (vertexIndex-1)/maxCols != maxLines-1 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex+maxCols))
            dic["down"].append(index)
            index+=1
    for vertexIndex in range(1,nbVertices-1):
        currentVertex = basicGraph.vertex(vertexIndex)
        if (vertexIndex-1)/maxCols != 0 :
            basicGraph.add_edge(currentVertex,basicGraph.vertex(vertexIndex-maxCols))
            dic["up"].append(index)
            index+=1
    

    # for key in dic :
    #     print(key,dic[key][0],dic[key][-1])

    # print(maxCols,maxLines)
    # print(basicGraph.get_edges())
    propertyGraph = basicGraph.new_edge_property("float")
    basicGraph.edge_properties["w"] = propertyGraph
    return basicGraph,dic

    #### UTILISER set_2d_array(a, pos=None)

def getCoords(vertexNumber,maxCols,maxLines):

    y = (vertexNumber-1)/maxCols 
    x = (vertexNumber-1)%maxCols 

    return int(x),int(y)-1
    

# def putEdgeWeight(basicGraph,distance,potential,disparity,labelNumber,maxCols,maxLines):
#     nbVertices = maxCols*maxLines+2 

#     for i in tqdm.tqdm(range(1,nbVertices-1)):
#         currentVertex = basicGraph.vertex(i)
#         x,y = getCoords(i,maxCols,maxLines)
#         edges = basicGraph.get_in_edges(currentVertex)
#         for edge in edges :
#             edgeNumber = edge[2]
#             vertexFrom = edge[0]
#             if vertexFrom ==0 :
#                 test = distance[disparity[x,y],x,y]
#                 basicGraph.edge_properties["w"][edge] = distance[disparity[x,y],x,y]
#             else :
#                 xFrom,yFrom = getCoords(vertexFrom,maxCols,maxLines)
#                 # print(xFrom,yFrom,x,y,)
#                 # print(disparity[x,y],disparity[xFrom,yFrom])
#                 # print(labelNumber)
#                 # print(np.shape(potential))
#                 basicGraph.edge_properties["w"][edge] = (potential[disparity[xFrom,yFrom],labelNumber,x,y] + potential[disparity[xFrom,yFrom],labelNumber,xFrom,yFrom])


#     puit = basicGraph.vertex(nbVertices-1)
#     edges = basicGraph.get_in_edges(puit)
#     for edge in edges :
#         edgeNumber = edge[2]
#         vertexFrom = edge[0]
#         x,y = getCoords(vertexFrom,maxCols,maxLines)
#         basicGraph.edge_properties["w"][edge] = distance[disparity[x,y],x,y]


def createCurrentPotent(potential,disparity,labelNumber):
    potent = np.zeros((np.shape(disparity)[0],np.shape(disparity)[1]))
    for i in range(len(potent)):
        for j in range(len(potent[0])):
            potent[i,j] = potential[disparity[i,j],labelNumber,i,j]
    return potent

def createCurrentDist(dataCost,disp):
    currentDataCost = np.zeros((np.shape(disp)[0],np.shape(disp)[1]))
    # print(np.shape(disp))
    for i in range(len(currentDataCost)):
        for j in range(len(currentDataCost[0])):
            currentDataCost[i,j] = dataCost[disp[i,j],i,j]
    return currentDataCost

def putEdgeWeightFast(basicGraph,dic,dataCost,potential,disparity,labelNumber,maxCols,maxLines):
    nbVertices = maxCols*maxLines+2
    
    pot= createCurrentPotent(potential,disparity,labelNumber)
    currentDist = createCurrentDist(dataCost,disparity)
    # print(np.shape(currentDist))
    # print(np.shape(pot))
    # print(np.shape(dic["source"]))

    aux = []

    aux.extend((currentDist[:,:]).flatten())
    # print("aux1",np.shape(aux))
    aux.extend((dataCost[labelNumber,:,:]).flatten())
    # print("aux2",np.shape(aux))
    aux.extend(np.transpose(pot[1:,:]+pot[-1:,:]).flatten())
    aux.extend(np.transpose(pot[1:,:]+pot[-1:,:]).flatten())
    aux.extend(np.transpose(pot[:,1:]+pot[:,:-1]).flatten())
    aux.extend(np.transpose(pot[:,1:]+pot[:,:-1]).flatten())
    # print("aux3",np.shape(aux))
    aux = np.array(aux)
    
    basicGraph.edge_properties["w"].set_2d_array(aux)




def graphCut(images,distance,H,potential,lambda_value,gamma_value,disparityBegin,maxIter = 50):
    maxCols = np.shape(images)[1]
    maxLines = np.shape(images)[2]
    nbVertices = maxCols*maxLines+2

    disparity = copy.deepcopy(disparityBegin)
    # Calculate data cosr :
    data_cost = float(lambda_value) * distance + gamma_value * H

    # Create graph :
    # basicGraph = createGraph(images)
    basicGraph,dic = createGraphFast(images)
    source = basicGraph.vertex(0)
    puit = basicGraph.vertex(nbVertices-1)

    # print(np.shape(distance))

    noChange = False
    count = 0
    energy = compute_energy(lambda_value,distance,gamma_value,H,disparity,potential,details=True)
    print("First energy is {}".format(energy))
    energyAux = 0
    while (not noChange) and count<maxIter :
        noChange = True 
        for labelNumber in range(len(images)):
            # print("Label Number {} and count {} and maxIter {}".format(labelNumber,count,maxIter))
            # if np.all(distance[:,:,labelNumber]>100) :
            # print(len(np.where(distance[:,:,labelNumber]>10)[0]),np.shape(distance)[1]*np.shape(distance)[2]*0.8)
            if len(np.where(distance[labelNumber,:,:]>10)[0])>np.shape(distance)[1]*np.shape(distance)[2]*0.5:
                continue
            # putEdgeWeight(basicGraph,distance,potential,disparity,labelNumber,maxCols,maxLines)
            putEdgeWeightFast(basicGraph,dic,data_cost,potential,disparity,labelNumber,maxCols,maxLines)
            # g = graph_tool.price_network(3000)
            # pos = graph_tool.draw.sfdp_layout(g)
            # graph_tool.draw.graph_draw(g, pos=pos, output="graph-draw-sfdp2.pdf")
            # pos = graph_tool.draw.sfdp_layout(basicGraph,eweight = basicGraph.edge_properties["w"] )
            # graph_tool.draw.graph_draw(basicGraph,pos = pos,output="graph{}.pdf".format(labelNumber))
            # print("weight put")
            residual = graph_tool.flow.boykov_kolmogorov_max_flow(basicGraph, source, puit, basicGraph.edge_properties["w"])
            # print("residual found")
            result = graph_tool.flow.min_st_cut(basicGraph, source, basicGraph.edge_properties["w"],residual)
            # print("OK")
            partition = result.get_array()
            # print(np.all(partition.get_array()))
            # print("partition",len(np.where(partition==False)[0]))
            # print(np.shape(partition))
            disparityAux = copy.deepcopy(disparity)
            
                
            indices = np.where(partition==True)
            indices2 = np.where(partition==False)
            for index in indices[0] :
                x,y = getCoords(index,maxCols,maxLines)
                disparityAux[x,y] = labelNumber
            energyAux = compute_energy(lambda_value,distance,gamma_value,H,disparityAux, potential)
            # print("New {} OLD {}".format(energyAux,energy))
            if energyAux<energy:
                # print("New {} OLD {}".format(energyAux,energy))
                disparity = copy.deepcopy(disparityAux)
                noChange = False
                # print("NICE")
                energy = copy.deepcopy(energyAux)


            for index in indices2[0] :
                x,y = getCoords(index,maxCols,maxLines)
                disparityAux[x,y] = labelNumber
            energyAux = compute_energy(lambda_value,distance,gamma_value,H,disparityAux, potential)
            # print("New {} OLD {}".format(energyAux,energy))
            if energyAux<energy:
                # print("New {} OLD {}".format(energyAux,energy))
                disparity = copy.deepcopy(disparityAux)
                noChange = False
                # print("NICE")
                energy = copy.deepcopy(energyAux)

        count+=1
    print("Last energy is {}".format(compute_energy(lambda_value,distance,gamma_value,H,disparity,potential,details=True)))
    print("Optimization took {} iterations".format(count))

    return disparity





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
    # H = calculateMap(images)
    # averageImage = []
    H = np.zeros((np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))

    print "Calculate cost distance"
    with open(os.path.join(inputPath,"pickleAux/distance.pck"),"rb") as f :
        distance = pickle.load(f)


    print("Normalize distance")  
    distance = np.array(normalize(distance))

    #Algorithm
    lambda_value = 0.1
    gamma_value = 1
    print "Begin minimisation"


    potential = calculatePotentialCost(images)
    # potential = np.zeros((np.shape(images)[0],np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))
    disparity = np.random.randint(0,len(images),(np.shape(images)[1],np.shape(images)[2]))
    # disparity = np.argmin(distance,axis = 0)

    


    height = np.shape(images)[1]
    width = np.shape(images)[2]
    numberOfSeparation = 5
    reducedHeight = height/(2*numberOfSeparation+1)
    reducedWidth = width/(2*numberOfSeparation+1)
    print(np.all(distance>1000,axis =0))
    print(np.where(np.all(distance>1000,axis =0)))
    # image = 
    imageShape = np.shape(images[0])
    height = imageShape[0]/(2*numberOfSeparation+1)
    for i in range(numberOfSeparation) :
        for j in tqdm.tqdm(range(numberOfSeparation)):
            print("We're at i,j:{},{}".format(i,j))
            disparityReduced = disparity[2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            imagesReduced = images[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            distanceReduced = distance[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            HReduced = H[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            potentialReduced = potential[:,:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            disparity_est = graphCut(imagesReduced,distanceReduced,HReduced,potentialReduced,lambda_value,gamma_value,disparityReduced)
            disparity[2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1] = disparity_est
            newImage = reconstructionImage(images,disparity)
            cv2.imwrite(os.path.join(inputPath,"output/MRFCOST{}{}.jpeg".format(i,j)),newImage)




    # newImage = reconstructionImage(images,disparity_est)
    
    
    # cv2.imwrite("output3/output/MRFCOST.jpeg",newImage)