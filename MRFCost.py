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
    
    
    new_cost[:-2,:,:] += msg_up[1:-1,:,:]
    
    new_cost[2:,:,:] += msg_down[1:-1,:,:]   
        
    new_cost[:,:-2,:] += msg_left[:,1:-1,:]   
    
    new_cost[:,2:,:] += msg_right[:,1:-1,:]
    
    belief = copy.deepcopy(new_cost)

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
    for i in tqdm.tqdm(range(1,len(data_cost)-1)):
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

    
    return energy
    

def compute_MAP_labeling(beliefs) :
    print("MAP")
    return beliefs.argmin(axis = 2)
    


def normalize_messages(msg_up, msg_down, msg_left, msg_right):
    print("normalize")
    shape = np.shape(msg_up)
    mean_msg_val = msg_up.mean(axis=2)
    print(np.shape(msg_up),np.shape(np.transpose(msg_up,(2,0,1))))
    msg_up = np.transpose(np.subtract(np.transpose(msg_up,(2,0,1)),mean_msg_val),(1,2,0))
    # for i in range(np.shape(msg_up)[2]):
        # msg_up[0:shape[0],0:shape[1],i] =  msg_up[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
        # msg_up[0:shape[0],0:shape[1],i] =  msg_up[0:shape[0],0:shape[1],i] /mean_msg_val[0:shape[0],0:shape[1]]
    
    
    shape = np.shape(msg_down)
    mean_msg_val = msg_down.mean(axis=2)
    msg_down = np.transpose(np.subtract(np.transpose(msg_down,(2,0,1)),mean_msg_val),(1,2,0))
    # for i in range(np.shape(msg_down)[2]):
        # msg_down[0:shape[0],0:shape[1],i] =  msg_down[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
        # msg_down[0:shape[0],0:shape[1],i] =  msg_down[0:shape[0],0:shape[1],i] /mean_msg_val[0:shape[0],0:shape[1]]
    
    shape = np.shape(msg_left)
    mean_msg_val = msg_left.mean(axis=2)
    msg_left = np.transpose(np.subtract(np.transpose(msg_left,(2,0,1)),mean_msg_val),(1,2,0))
    # for i in range(np.shape(msg_left)[2]):
        # msg_left[0:shape[0],0:shape[1],i] =  msg_left[0:shape[0],0:shape[1],i] - mean_msg_val[0:shape[0],0:shape[1]]
        # msg_left[0:shape[0],0:shape[1],i] =  msg_left[0:shape[0],0:shape[1],i] /mean_msg_val[0:shape[0],0:shape[1]]
    
    shape = np.shape(msg_right)
    mean_msg_val = msg_right.mean(axis=2)
    msg_right = np.transpose(np.subtract(np.transpose(msg_right,(2,0,1)),mean_msg_val),(1,2,0))
    # for i in range(np.shape(msg_right)[2]):
        # msg_right[:,:,i] =  msg_right[:,:,i] - mean_msg_val[:,:]
        # msg_right[0:shape[0],0:shape[1],i] =  msg_right[0:shape[0],0:shape[1],i] /mean_msg_val[0:shape[0],0:shape[1]]
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
    aux_up[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[2:,1:-1,:]+msg_left_prev[1:-1,2:,:]+msg_right_prev[1:-1,:-2,:]+data_cost[1:-1,1:-1,:])
    aux_down[1:-1,1:-1,:] = copy.deepcopy(msg_down_prev[:-2,1:-1,:]+msg_left_prev[1:-1,2:,:]+msg_right_prev[1:-1,:-2,:]+data_cost[1:-1,1:-1,:])
    aux_left[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[2:,1:-1,:]+msg_down_prev[:-2,1:-1,:]+msg_left_prev[1:-1,2:,:]+data_cost[1:-1,1:-1,:])
    aux_right[1:-1,1:-1,:] = copy.deepcopy(msg_up_prev[2:,1:-1,:]+msg_down_prev[:-2,1:-1,:]+msg_right_prev[1:-1,:-2,:]+data_cost[1:-1,1:-1,:])
    


    print("update message")
    for i in tqdm.tqdm(range(num_disp_vals)):
        msg_up[1:-1,1:-1,i] = np.min(aux_up[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]\
            +np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[2:,1:-1,:],axis=2)

        msg_down[1:-1,1:-1,i] = np.min(aux_down[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]+\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[:-2,1:-1,:],axis=2)


        msg_left[1:-1,1:-1,i] = np.min(aux_left[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:] +\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,2:,:],axis=2)

        msg_right[1:-1,1:-1,i] = np.min(aux_right[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:]+\
            np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,0:-2,:],axis=2)



    
    # for i in range(num_disp_vals):
    #     msg_up[1:-1,1:-1,i] = np.min(aux_up[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:],axis=2)

    #     msg_down[1:-1,1:-1,i] = np.min(aux_down[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:],axis=2)

    #     msg_right[1:-1,1:-1,i] = np.min(aux_right[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:],axis=2)


    #     msg_left[1:-1,1:-1,i] = np.min(aux_left[1:-1,1:-1,:] + np.transpose(np.transpose(potentialTotal[i]),(1,0,2))[1:-1,1:-1,:],axis=2)

    

    return msg_up,msg_down,msg_left,msg_right
 


def calculatePotentialCost(images):
    potentialTotal = []
    potential = np.zeros(np.shape(images)[:-1])

    for i in tqdm.tqdm(range(np.shape(images)[0])):
        potential[:,:,:] = np.linalg.norm(np.subtract(images[:,:,:,:],images[i,:,:,:]),axis=3)
        potentialTotal.append(copy.deepcopy(potential))

    return np.array(potentialTotal)


def stereo_belief_propagation(images,distance, H, potential,lambda_value, gamma_value,outputPath):
    num_disp_values = np.shape(images)[0] #number of disparity values
    tau             = 15 
    num_iterations  = 100 #number of iterations
    height, width = np.shape(images[0])[0],np.shape(images[0])[1]
    

    data_cost = np.zeros((np.shape(distance)[1],np.shape(distance)[2],np.shape(distance)[0]))
    for i in range(len(distance)):
        data_cost[:,:,i] = float(lambda_value) * distance[i,:,:] + gamma_value * H[i,:,:]


    # potentialTotal = calculatePotentialCost(images)
    potentialTotal = copy.deepcopy(potential)
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
        beliefs = compute_belief(data_cost, msg_up, msg_down, msg_left, msg_right)
        
        #compute MAP disparities
        #disparity: a 2D array of size height x width the disparity value of each 
        #  pixel; the disparity values range from 0 till num_disp_value - 1
        disparity = compute_MAP_labeling(beliefs)

        # if iter%10 == 0 :
            
        newImage = reconstructionImage(images,disparity)

        # if iter%5 == 0:
        #     if iter >2 :
        #         cv2.imwrite(os.path.join(outputPath,"output/Diff{}.jpeg".format(iter)),np.linalg.norm(np.abs(newImage-old_image),axis=2))
        #         print(np.sum(np.abs(newImage-old_image)))
        #     old_image = copy.deepcopy(newImage)
        #     cv2.imwrite(os.path.join(outputPath,"output/MRFCOST{}.jpeg".format(iter)),newImage)
        #     cv2.imwrite(os.path.join(outputPath,"output/Disparity{}.jpeg".format(iter)),np.array(disparity)*10)
        #compute MRF energy   
        energy.append(compute_energy(data_cost,float(lambda_value),distance,gamma_value,H, disparity, potentialTotal))
        # print("Total Energy: ",energy[-1])

        if len(energy)>2 and energy[-1]-energy[-2]==0:
            break
        
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
    for i in tqdm.tqdm(range(len(distance[0]))):
        for j in range(len(distance[0][0])):

            Inf = True
            for k in range(len(distance)):
                if not isinf(distance[k][i][j]) or distance[k][i][j]< 10:
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
                medianImage[i,j,k] = np.median(listValue)

    possibleResult = np.abs(np.subtract(images,medianImage))
    print("POSSIBLE",possibleResult[np.where(possibleResult>1)])

    MAD = np.median(possibleResult,axis = 3)
    print(MAD[np.where(MAD>3)])
    H = np.where(MAD<10,np.linalg.norm(possibleResult),0)
    print(H[np.where(H>0)])
    return H



def reconstructionImage(images,disparity):
    new_image = np.zeros(np.shape(images[0]))
    disparity = disparity.astype(int)
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

    inputPath = "outputVideoBlur"
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
    # H = calculateMap(images)
    # averageImage = []
    H = np.zeros((np.shape(images)[0],np.shape(images)[1],np.shape(images)[2]))
   
    print "Calculate cost distance"
    with open(os.path.join(inputPath,"pickleAux/distance.pck"),"rb") as f :
        distance = pickle.load(f)
    print(np.shape(distance))

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
    potential = calculatePotentialCost(images)





    height = np.shape(images)[1]
    width = np.shape(images)[2]
    numberOfSeparation = 10
    reducedHeight = height/(2*numberOfSeparation+1)
    reducedWidth = width/(2*numberOfSeparation+1)

    
    # image = 
    imageShape = np.shape(images[0])
    disparity = np.zeros(imageShape[:2])
    height = imageShape[0]/(2*numberOfSeparation+1)
    for i in range(numberOfSeparation) :
        for j in range(numberOfSeparation):
            print("We're at i,j:{},{}".format(i,j))
            # disparityReduced = disparity[2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            imagesReduced = images[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            distanceReduced = distance[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            HReduced = H[:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            potentialReduced = potential[:,:,2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]
            
            disparity_est, energy = stereo_belief_propagation(imagesReduced,distanceReduced,HReduced,potentialReduced,lambda_value,gamma_value,inputPath)
            print("The energy is {}".format(energy))
            print(disparity)
            disparity[2*i*reducedHeight:(2*i+3)*reducedHeight+1,2*j*reducedWidth:(2*j+3)*reducedWidth+1]= copy.deepcopy(disparity_est)
            newImage = reconstructionImage(images,disparity)
            cv2.imwrite(os.path.join(inputPath,"output/MRFCOST{}{}.jpeg".format(i,j)),newImage)

    # disparity_est, energy = stereo_belief_propagation(images,distance,H,lambda_value,gamma_value,inputPath)
    # disparity_est = np.argmin(distance,axis = 0)

    print(disparity_est)
    print(np.histogram(disparity_est))
    print(np.shape(disparity_est))

    import matplotlib.pyplot as plt

    plt.plot(energy)
    plt.show()


    # newImage = reconstructionImage(images,disparity_est)
    
    
    # cv2.imwrite("output3/output/MRFCOST.jpeg",newImage)