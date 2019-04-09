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
def compute_belief(data_cost, msg_up, msg_down, msg_left, msg_right):
    print("compute_belief")
    
    new_cost = copy.deepcopy(data_cost)
    
    
    new_cost[:-1,:,:] += msg_up[1:,:,:]
    
    new_cost[1:,:,:] += msg_down[:-1,:,:]   
        
    new_cost[:,:-1,:] += msg_left[:,1:,:]   
    
    new_cost[:,1:,:] += msg_right[:,:-1,:]
    
    belief = new_cost

    # print(belief)
    
    return belief
    
# compute_energy(float(lambda_value),distance,gamma_value,H, disparity, potential_cost_up,potential_cost_down,potential_cost_left,potential_cost_right))
def compute_energy(data_cost,lambda_value,distance,gamma_value,H,disparity, potential):
    print("compute_energy")
    energy = 0
    energyDistance = 0
    energyMAD = 0
    energyPotential = 0
    energyDataCost = 0
    for i in range(1,len(data_cost)-1):
        for j in range(1,len(data_cost[i])-1):
            if isinf(data_cost[i,j,disparity[i,j]]):

                continue
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
            aux4 = data_cost[i,j,disparity[i,j]]
            # print(aux)
            if isnan(aux1) or isinf(aux1):
                aux1 = 0
            if isnan(aux2) or isinf(aux2) :
                aux2 = 0
            if isnan(aux3) or isinf(aux3):
                aux3 = 0
            if isnan(aux4) or isinf(aux4):
                aux4 = 0

            if aux1<0 or aux2<0 or aux3<0 or aux4<0 :
                print("NEGATIF",i,j)
            # if aux<0 : print(data_cost[i,j,disparity[i,j]],potentialUp[disparity[i,j],disparity[i+1,j],i,j] )
            energy+=(aux2 +aux4)
            energyDistance +=aux1
            energyMAD+=aux3
            energyPotential+=aux2
            energyDataCost+=aux4


    print("Energy Distance", energyDistance)
    print("Energy MAD", energyMAD)
    print("energyPotential",energyPotential)
    print("Energy Data Cost", energyDataCost)
                    
    # energy = np.sum(data_cost[:,:,disparity[:,:]])

    # print(energy)
    # left_energy = np.sum(potentialLeft)
    
    # right_energy = np.sum(np.where(disparity[1:shape[0]-1,1:shape[1]-1] != disparity[1:shape[0]-1,2:shape[1]],1,0))
    
    # up_energy = np.sum(np.where(disparity[1:shape[0]-2,1:shape[1]-1] != disparity[1:shape[0]-1,1:shape[1]-1],1,0))
    
    # down_energy = np.sum(np.where(disparity[2:shape[0],1:shape[1]-1] != disparity[1:shape[0]-1,1:shape[1]-1],1,0))
    
    # energy += lambda_value*(left_energy + up_energy + right_energy + down_energy)
    
    
    return energy
    
    
# def compute_data_cost(img_left, img_right, num_disp_vals, tau) : 
#     print("compute_data_cost")
#     shape = np.shape(img_left)
#     data_cost = np.zeros((shape[0],shape[1],num_disp_vals))
#     tau_comp = (np.ones((shape[0],shape[1],num_disp_vals),dtype = int)*int(tau))
    
#     for i in range(num_disp_vals) :
#         if i!=0 :
#             x = np.abs(img_left[:,i:]-img_right[:,:-i])
#             y = tau_comp[:,i:,i]
#             data_cost[:,i:,i] = np.where(x<y,x,y)
#             # data_cost[:,i:,i] = np.min(np.abs(img_left[:,i:]-img_right[:,:-i]),tau_comp[:,:,i])
#         else :
#             # data_cost[:,:,0] = np.min(np.abs(img_left[:,:]-img_right[:,:]),tau_comp[:,:,0])
#             x = np.abs(img_left[:,:]-img_right[:,:])
#             y = tau_comp[:,:,0]
#             data_cost[:,:,0] = np.where(x<y,x,y)
#     return data_cost
    


def compute_MAP_labeling(beliefs) :
    print("MAP")
    return beliefs.argmin(axis = 2)
    


def normalize_messages(msg_up, msg_down, msg_left, msg_right):
    print("normalize")
    shape = np.shape(msg_up)
    mean_msg_val = msg_up.mean(axis=2)
    for i in range(np.shape(msg_up)[2]):
        msg_up[0:shape[0],0:shape[1],i] =  msg_up[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_down)
    mean_msg_val = msg_down.mean(axis=2)
    for i in range(np.shape(msg_down)[2]):
        msg_down[0:shape[0],0:shape[1],i] =  msg_down[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_left)
    mean_msg_val = msg_left.mean(axis=2)
    for i in range(np.shape(msg_left)[2]):
        msg_left[0:shape[0],0:shape[1],i] =  msg_left[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_right)
    mean_msg_val = msg_right.mean(axis=2)
    for i in range(np.shape(msg_right)[2]):
        msg_right[0:shape[0],0:shape[1],i] =  msg_right[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
        
    return msg_up, msg_down, msg_left, msg_right
    
# def update_messages(msg_up_prev, msg_down_prev, msg_left_prev, msg_right_prev, data_cost, potentialUp,potentialDown,potentialLeft,potentialRight):
def update_messages(msg_up_prev, msg_down_prev, msg_left_prev, msg_right_prev, data_cost, potentialTotal):
    width,height,nbValue = np.shape(msg_up_prev)

    #Initialisation des messages precedents :
    msg_up = copy.deepcopy(msg_up_prev)
    msg_down = copy.deepcopy(msg_down_prev)
    msg_left = copy.deepcopy(msg_left_prev)
    msg_right = copy.deepcopy(msg_right_prev)
    
    num_disp_vals = np.shape(msg_up)[2]
  
    # Calcul de la somme des messages precedents et du cout naturelle
    # en fonction du pixel choisi et du type de message a envoyer, les msg ne sont pas definies partout :

    aux_up=np.zeros(np.shape(msg_up))
    aux_down=np.zeros(np.shape(msg_up))
    aux_right=np.zeros(np.shape(msg_up))
    aux_left=np.zeros(np.shape(msg_up))

    # print(np.histogram(data_cost[np.where(data_cost<10000)]))
    aux_up[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[0:-2,1:-1,:]+msg_left_prev[1:-1,0:-2,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:])
    aux_down[1:-1,1:-1,:] = copy.deepcopy(msg_down_prev[2:,1:-1,:]+msg_left_prev[1:-1,:-2,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:])
    aux_left[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[0:-2,1:-1,:]+msg_down_prev[2:,1:-1,:]+msg_left_prev[1:-1,0:-2,:]+data_cost[1:-1,1:-1,:])
    aux_right[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[0:-2,1:-1,:]+msg_down_prev[2:,1:-1,:]+msg_right_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:])
    
    # min_up = (aux_up+lambda_value).min(axis = 2)
    # min_down = (aux_down+lambda_value).min(axis = 2)
    # min_right =(aux_right+lambda_value).min(axis = 2)
    # min_left = (aux_left+lambda_value).min(axis = 2)
    # print(np.shape(aux_down))
    # print(np.histogram(aux_up[:,:,1]))

    # print(np.shape(aux_up),np.shape(potentialUp[0]).reshape(width,height,nbValue),np.shape(msg_up))
    # print(np.shape(np.transpose(np.transpose(potentialDown[0]),(1,0,2))),np.shape(aux_up))
    for i in range(num_disp_vals):
        msg_up[1:-1,1:-1,i] = np.min(aux_up[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]\
            +np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[0:-2,1:-1,:],axis=2)
        # msg_up[1:-1,1:-1,i] = np.min(aux_up[1:-1,1:-1,:] + np.transpose(np.transpose(potentialUp[i]),(1,0,2))[1:-1,1:-1,:]\
            # ,axis=2)

        # msg_up[1:-1,1:-1,i] = np.where(np.min(aux_up[1:-1,1:-1,:],axis=2)+1<aux_up[1:-1,1:-1,i],\
            # np.min(aux_up[1:-1,1:-1,:],axis=2),aux_up[1:-1,1:-1,i])
        # msg_up[1:-1,1:-1,i] = np.min(aux_up[1:-1,1:-1,:],axis=2)
        msg_down[1:-1,1:-1,i] = np.min(aux_down[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]+\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[2:,1:-1,:],axis=2)
        # msg_down[1:-1,1:-1,i] = np.min(aux_down[1:-1,1:-1,:] + np.transpose(np.transpose(potentialDown[i]),(1,0,2))[1:-1,1:-1,:]\
            # ,axis=2)
        # msg_down[1:-1,1:-1,i] = np.where(np.min(aux_down[1:-1,1:-1,:],axis=2)+1<aux_down[1:-1,1:-1,i],\
            # np.min(aux_down[1:-1,1:-1,:],axis=2),aux_down[1:-1,1:-1,i])
        # msg_down[1:-1,1:-1,i] = np.min(aux_down[1:-1,1:-1,:],axis=2)
        msg_right[1:-1,1:-1,i] = np.min(aux_right[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]+\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,0:-2,:],axis=2)
        # msg_right[1:-1,1:-1,i] = np.min(aux_right[1:-1,1:-1,:] + np.transpose(np.transpose(potentialRight[i]),(1,0,2))[1:-1,1:-1,:]\
            # ,axis=2)
        # msg_right[1:-1,1:-1,i] = np.where(np.min(aux_right[1:-1,1:-1,:],axis=2)+1<aux_right[1:-1,1:-1,i],\
            # np.min(aux_right[1:-1,1:-1,:],axis=2),aux_right[1:-1,1:-1,i])
        # msg_right[1:-1,1:-1,i] = np.min(aux_right[1:-1,1:-1,:],axis = 2)

        # msg_left[1:-1,1:-1,i] = np.where(np.min(aux_left[1:-1,1:-1,:],axis=2)+1<aux_left[1:-1,1:-1,i],\
            # np.min(aux_left[1:-1,1:-1,:],axis=2),aux_left[1:-1,1:-1,i])
        msg_left[1:-1,1:-1,i] = np.min(aux_left[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:] +\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,2:,:],axis=2)
        # msg_left[1:-1,1:-1,i] = np.min(aux_left[1:-1,1:-1,:] + np.transpose(np.transpose(potentialLeft[i]),(1,0,2))[1:-1,1:-1,:]\
            # ,axis=2)
        # msg_left[1:-1,1:-1,i] = np.min(aux_left[1:-1,1:-1,:],axis=2)
    
    # msg_up = np.array(msg_up)
    # print(np.histogram(msg_up[:,:,1]))
    return msg_up,msg_down,msg_right,msg_left
 


def calculatePotentialCost(images):
    potentialTotal = []
    # potentialUpTotal = []
    # potentialLeftTotal = []
    # potentialRightTotal = []



    potential = np.zeros(np.shape(images)[:-1])
    # potentialDown = np.zeros(np.shape(images)[:-1])
    # potentialLeft =  np.zeros(np.shape(images)[:-1])
    # potentialRight =  np.zeros(np.shape(images)[:-1])

    for i in tqdm.tqdm(range(np.shape(images)[0])):
        # print(np.shape(images[:,1:-1,1:-1:,:]),np.shape(images[i,0:-2,1:-1,:]))
        # test = np.subtract(images[:,1:-1,1:-1:,:],images[i,0:-2,1:-1,:])
        # print(np.shape(test))
        # a = np.linalg.norm(test,axis=3)
        # print(np.shape(a))
        # aux = np.linalg.norm(np.subtract(images[:,1:-1,1:-1:,:],images[i,0:-2,1:-1,:]),axis=3)
        # print(np.shape(aux),np.shape(potentialUp))
        # potentialUp[:,1:-1,1:-1] = aux

      
        # potentialUp[:,1:-1,1:-1] = np.linalg.norm(np.subtract(images[:,1:-1,1:-1:,:],images[i,1:-1,1:-1,:]),axis=3)
        # potentialDown[:,1:-1,1:-1] = np.linalg.norm(np.subtract(images[:,1:-1,1:-1:,:],images[i,:-2,1:-1,:]),axis=3)
        # potentialLeft[:,1:-1,1:-1] = np.linalg.norm(np.subtract(images[:,1:-1:,1:-1,:],images[i,1:-1,:-2,:]),axis=3)
        # potentialRight[:,1:-1,1:-1] = np.linalg.norm(np.subtract(images[:,1:-1:,1:-1,:],images[i,1:-1,2:,:]),axis=3)

        # potentialDownTotal.append(copy.deepcopy(potentialDown))
        # potentialUpTotal.append(copy.deepcopy(potentialUp))
        # potentialLeftTotal.append(copy.deepcopy(potentialLeft))
        # potentialRightTotal.append(copy.deepcopy(potentialRight))


        potential[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)
        # potentialDown[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)
        # potentialLeft[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)
        # potentialRight[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)

        potentialTotal.append(copy.deepcopy(potential))
        # potentialUpTotal.append(copy.deepcopy(potentialUp))
        # potentialLeftTotal.append(copy.deepcopy(potentialLeft))
        # potentialRightTotal.append(copy.deepcopy(potentialRight))


        

        
    # print(np.where(potentialRightTotal[0][0,:,:]!=0))
    # print(potentialRightTotal[0][0][np.where(potentialRightTotal[0][0,:,:]!=0)])
    # print(np.where(potentialRightTotal[1][0,:,:]!=0,potentialRightTotal[1][0,:,:]))
    return np.array(potentialTotal)
# For convenience do not compute the messages that are sent from pixels that are on the boundaries of the image. 
# Compute the messages only for the pixels with coordinates (y,x) = ( 2:(height-1) , 2:(width-1) )
## CONCATENATION :

def stereo_belief_propagation(images,distance, H,lambda_value, gamma_value,outputPath):
    num_disp_values = np.shape(images)[0] #number of disparity values
    # images = images.reshape(np.shape(images)[1],np.shape(images)[2],np.shape(images)[3],num_disp_values)
    tau             = 15 
    num_iterations  = 100 #number of iterations
    height, width = np.shape(images[0])[0],np.shape(images[0])[1]
    


    #compute the data cost term
    #data_cost: a 3D array of size height x width x num_disp_value; each
    #  element data_cost(y,x,l) is the cost of assigning the label l to pixel 
    #  p = (y,x)

    data_cost = np.zeros((np.shape(distance)[1],np.shape(distance)[2],np.shape(distance)[0]))
    for i in range(len(distance)):
        data_cost[:,:,i] = float(lambda_value) * distance[i,:,:] + gamma_value * H[i,:,:]

    # print(np.shape(data_cost))
    # print(np.histogram(distance[np.where(distance<100)]))
    # print(np.histogram(data_cost[np.where(data_cost<100)]))

    # for i in range(15):
        # print(data_cost[250,500,i])
    # data_cost = float(lambda_value) * distance 


    # potential_cost_up,potential_cost_down,potential_cost_right,potential_cost_left = calculatePotentialCost(images)
    potentialTotal = calculatePotentialCost(images)
    print(np.shape(potentialTotal))
    #allocate memory for storing the energy at each iteration
    energy = []
    
    #Initialize the messages at iteration 0 to all zeros
    
    #msg_up : a 3D array of size height x width x num_disp_value; each vector
    #  msg_up(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel up with coordinates q = (y-1,x)
    msg_up    = np.zeros((height, width, num_disp_values))
    #msg_down : a 3D array of size height x width x num_disp_value; each vector
    #  msg_down(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel down with coordinates q = (y+1,x)
    msg_down  = np.zeros((height, width, num_disp_values))
    #msg_left : a 3D array of size height x width x num_disp_value; each vector
    #  msg_left(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel left with coordinates q = (y,x-1)
    msg_left  = np.zeros((height, width, num_disp_values))
    #msg_right : a 3D array of size height x width x num_disp_value; each vector
    #  msg_right(y,x,1:num_disp_values) is the message vector that the pixel 
    #  p = (y,x) will send to the pixel right with coordinates q = (y,x+1)
    msg_right = np.zeros((height, width, num_disp_values))
    
    for iter in range(num_iterations):
        print("iter", iter)
        #update messages
        msg_up, msg_down, msg_left, msg_right = update_messages(msg_up, msg_down, msg_left, msg_right, data_cost,potentialTotal)
    
        msg_up, msg_down, msg_left, msg_right = normalize_messages(msg_up, msg_down, msg_left, msg_right)
        
        #compute  beliefs
        #beliefs: a 3D array of size height x width x num_disp_value; each
        #  element beliefs(y,x,l) is the belief of pixel p = (y,x) taking the
        #  label l
        beliefs = compute_belief( data_cost, msg_up, msg_down, msg_left, msg_right )
        
        #compute MAP disparities
        #disparity: a 2D array of size height x width the disparity value of each 
        #  pixel; the disparity values range from 0 till num_disp_value - 1
        disparity = compute_MAP_labeling(beliefs)

        # if iter%10 == 0 :
            
        newImage = reconstructionImage(images,disparity)
        if iter >2 :
            cv2.imwrite(os.path.join(outputPath,"output/Diff{}.jpeg".format(iter)),np.linalg.norm(np.abs(newImage-old_image),axis=2))
            print(np.sum(np.abs(newImage-old_image)))
        old_image = copy.deepcopy(newImage)
        cv2.imwrite(os.path.join(outputPath,"output/MRFCOST{}.jpeg".format(iter)),newImage)
        
        #compute MRF energy   
        energy.append(compute_energy(data_cost,float(lambda_value),distance,gamma_value,H, disparity, potentialTotal))
        print("Total Energy: ",energy[-1])
        # if len(energy)>2 and abs(energy[-1]-energy[-2])/energy[-2]<0.000001:
            # break
        
    # disparity = (disparity*(256/num_disp_values)).astype("uint8")
    # cv2.imshow("DISPARITY ITER {}".format(iter),disparity)
    return disparity,energy


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
    # M = np.zeros(np.shape(averageImage))

    # for i in tqdm.tqdm(range(len(averageImage))):
    #     for j in range(len(averageImage[i])):
    #         for k in range(len(averageImage[i][j])):
    #             nbImage = 0
    #             for imageindex in range(len(images)): # Inverser les deux for pour etre au point
    #                 if images[imageindex][i,j,k] != None :
    #                     nbImage+=1
    #                     M[i,j,k]+=(images[imageindex][i,j,k]-averageImage[i,j,k])**2
    #             if nbImage !=0 :
    #                 M[i,j,k] = M[i,j,k]/nbImage
    #             else :
    #                 M[i,j,k] = 0


    # sigma = np.linalg.norm(M,axis = 2)

    # MAD = np.subtract(images,M)
    # H = np.zeros((np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))
    # for i in tqdm.tqdm(range(len(images))) :
    #     for j in range(len(images[i])):
    #         for k in range(len(images[i][j])):
    #             if sigma[j,k]<10 :
    #                 H[i,j,k] = np.linalg.norm(MAD[i,j,k])


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
    print("POSSIBLE",possibleResult[np.where(possibleResult>1)])

    MAD = np.median(possibleResult,axis = 3)
    print(MAD[np.where(MAD>3)])
    H = np.where(MAD<10,np.linalg.norm(possibleResult),0)
    print(H[np.where(H>0)])
    return H
    
            
            


    return H



def reconstructionImage(images,disparity):
    new_image = np.zeros(np.shape(images[0]))
    print(np.shape(images))
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
        # images[-1]= putNone(images[-1])
    images = np.array(images)


    # averageImage = calculateAverageImage(images)
    # cv2.imwrite(os.path.join(inputPath,"output","avg.jpg"),averageImage)
    
    print "Calculate cost H"
    H = calculateMap(images)
    averageImage = []
    H = np.zeros((np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))

    print "Calculate cost distance"
    with open(os.path.join(inputPath,"pickleAux/distance.pck"),"rb") as f :
        distance = pickle.load(f)


    # distance = np.array(distance)


    print("Normalize distance")
    # 
    # 
    # distance = np.array(distance).reshape(np.shape(distance)[1],np.shape(distance)[2],np.shape(distance)[0])   
    distance = np.array(normalize(distance))

    #estimate the disparity map with the Max-Product Loopy Belief Propagation
    #Algorithm
    lambda_value = 100
    gamma_value = 0
    print "Begin minimisation"
    disparity_est, energy = stereo_belief_propagation(images,distance,H,lambda_value,gamma_value,inputPath)
    # disparity_est = np.argmin(distance,axis = 0)

    print(disparity_est)
    print(np.histogram(disparity_est))
    print(np.shape(disparity_est))
    # figure(3); clf(3);
    # imagesc(disparity_est); colormap(gray)
    # title('Disparity Image');
    # cv2.imshow("FINAL",disparity_est)
    # cv2.imshow("Ground TRUTH",disparity)
    # figure(4); clf(4);
    # plot(1:length(energy),energy)
    # title('Energy per iteration')
    import matplotlib.pyplot as plt

    plt.plot(energy)
    plt.show()


    # newImage = reconstructionImage(images,disparity_est)
    
    
    # cv2.imwrite("output3/output/MRFCOST.jpeg",newImage)