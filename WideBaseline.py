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
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
import spectral
#========#========#========#========#========#========#========#========#========#========#========#========
# Constants
#========#========#========#========#========#========#========#========#========#========#========#========

R = 50
gamma = 10
heightStep = 20
widthStep = 20
ratioGlob = 0.6
datasetNumber = 2
# datasetNUmber = ""

# heightStep = 20
# widthStep = 20
lambdaC = 1
lambdaB = 1e15
lambdas = 0.1

#========#========#========#========#========#========#========#========#========#========#========#========
# DEBUGGING TRIES :
#========#========#========#========#========#========#========#========#========#========#========#========
# ptsA.append(points[key[0]][indexPoint1])
# ptsB.append(points[key[1]][indexPoint2])


# Simple Shifting

dico1 = {}
pts1 = {}
dico1[0,1] = [[0,0],[1,1],[2,2]]
pts1[0] = [[1,1],[4,4],[3,4]]
pts1[1] = [[2,2],[4,4],[3,4]]

dico2 = {}
pts2 = {}
dico2[0,1] = [[0,0]]
pts2[0] = [[10,10]]
pts2[1] = [[150,150]]
# Homographie



#========#========#========#========#========#========#========#========#========#========#========#========
#Detection Code :
#========#========#========#========#========#========#========#========#========#========#========#========
# For features detection/description without threading
def detect_and_describe(image, width = 600):
    image = np.array(image)
    # Get Width & Height of the image
    # print(image)
    H, W = image.shape[:2]

    # widthFactor = 1.0 * width / image.shape[1]

    # Resize & grayify image
    # S = np.array([[widthFactor, 0.0, 0.0], [0.0, widthFactor, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    S = 1
    # small_image = cv2.warpPerspective(image, S, (int(W * widthFactor), int(H * widthFactor)))
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

    return kps, descriptors, S,kpscv




def get_features(images, width=800):
    """ Computes the features of each image using multitrheading
    Parameters
    ==========
    images: (list) - The list of images we want to compute the features
    width: (int) - The width of the image after resize (the smaller, the faster) [optional - default=800]
    Returns
    =======
    images: (list) - The list of images with the key "features", which contains the features
    """


    for image in images:
        image["features"] = detect_and_describe(image["image"], width)

    return images


def findNeighboursIndices(ref, pointsA,pointsB):
    listIndexNeighbours = []
    listNeighboursA = []
    listNeighboursB = []
    for i in range(len(pointsA)):
        norm = np.linalg.norm(ref-pointsA[i])
        if norm<R and norm>0 :
            listIndexNeighbours.append(i)
            listNeighboursA.append(pointsA[i])
            listNeighboursB.append(pointsB[i])


    return listNeighboursA,listNeighboursB,listIndexNeighbours

def featureMatching(featuresA,featuresB,imageA,imageB):
    
    shape = np.shape(imageA)
    kpsA, descriptorA, SA,kpscvA = featuresA
    kpsB, descriptorB, SB,kpscvB = featuresB


    matcher = cv2.DescriptorMatcher_create("FlannBased")
    ratio=ratioGlob
    matches = matcher.knnMatch(descriptorA, descriptorB, 2) 

    matches = filter(lambda m: len(m) == 2 and m[0].distance < m[1].distance * ratio, matches)
    
    
    # res = cv2.drawMatchesKnn(imageA,kpscvA,imageB,kpscvB,matches,None) 
    # plt.imshow(res)
    # plt.show()
    matchesAux = copy.deepcopy(matches)
    matches = [(m[0].trainIdx, m[0].queryIdx) for m in matches]
    



    ptsA = np.array(np.float32([kpsA[k] for (_, k) in matches]))
    ptsB = np.array(np.float32([kpsB[k] for (k, _) in matches]))
    indexA = np.array(np.float32([k for (_, k) in matches]))
    indexB = np.array(np.float32([k for (k, _) in matches]))


    # Calculate Homography
    listCommonA = outlier_rejection(ptsA,ptsB)
    listCommonA.sort()
    # listCommon = outlier_rejection(ptsA,ptsB)
    listCommonB = outlier_rejection(ptsB,ptsA)

    listCommonB.sort()
    # print(np.shape(listCommonB))
    # matches2 = [matchesAux[k] for k in listCommonB]
    # print(np.shape(matches2))
    # res = cv2.drawMatchesKnn(imageA,kpscvA,imageB,kpscvB,matches2,None) 
    # plt.imshow(res)
    # plt.show()
    # listCommon = np.union1d(listCommonA,listCommonB)
    listCommon = np.intersect1d(listCommonA,listCommonB)
    matches2 = [matchesAux[k] for k in listCommon]
    res = cv2.drawMatchesKnn(imageA,kpscvA,imageB,kpscvB,matches2,None) 
    plt.imshow(res)
    for k in listCommon :
        print(ptsA[k])
        plt.plot([ptsA[k][0],ptsB[k][0]+shape[1]],[ptsA[k][1],ptsB[k][1]],linewidth = 1)
    plt.show()
    listCommon.sort()
    print(listCommon)
    listeCouple = []
    for k in listCommon :
        k = int(k)
        listeCouple.append([indexA[k],indexB[k]])

    return np.array(listeCouple)

def get_features(image):
    kpsA, descriptorA, SA,kpscv = detect_and_describe(image)
    return [kpsA,descriptorA,SA,kpscv]


def getCommonFeatures(images):
    dico = {}
    points = {}
    features = {}

    for i in range(len(images)):
        features[i] = get_features(images[i])
        points[i] = features[i][0]

    for i in range(len(images)-1):
        for k in range(i+1,len(images)):
            print("HEHE")
            dico[i,k] = []
            listCommon = featureMatching(features[i],features[k],images[i],images[k])
            for index in listCommon :
                dico[i,k].append([index[0],index[1]])

    return dico,points





def outlier_rejection(ptsA,ptsB):
    # print("======================================")
    # print("SIMPLIFIED OUTLIER REJECTION TO CHANGE")
    # print("======================================")
    listCommon = []
    for point in tqdm.tqdm(ptsA) :
        listNeighboursA,listNeighboursB,listIndexNeighbours = findNeighboursIndices(point,ptsA,ptsB)

        # for i in range(len(listNeighboursA)):
            # listCommon.append(listIndexNeighbours[i])
        if len(listNeighboursA)>4 and len(listNeighboursB)>4:
            h, status = cv2.findHomography(np.array(listNeighboursA), np.array(listNeighboursB))
            # h = get_perspective_transform_matrix(listNeighboursA,listNeighboursB)
            if not np.any(status):
                continue
            for i in range(len(listNeighboursA)):
                if np.linalg.norm(apply_homography(h,listNeighboursA[i])-listNeighboursB[i])<gamma :
                    if listIndexNeighbours[i] not in listCommon : 
                        listCommon.append(listIndexNeighbours[i])
        else :
            for i in range(len(listNeighboursA)):
               
                if listIndexNeighbours[i] not in listCommon : 
                    listCommon.append(listIndexNeighbours[i])
               
    return listCommon


def apply_homography(homography,point):
    newPoint = [point[0],point[1],0]
    result = np.dot(homography,newPoint)
    return result[:2]

#========#========#========#========#========#========#========#========#========#========#========#========
#OPTIMISATION CODE 
#========#========#========#========#========#========#========#========#========#========#========#========

def create_grid(images,heightStep = heightStep,widthStep = widthStep):
    nbImages,height,width = np.shape(images)[:3]

    V = []
    for i in range(heightStep+1) :
        for j in range(widthStep+1):
            V.append(i*float(height-1)/(heightStep))
            V.append(j*float(width-1)/(widthStep))
    Vfinal = []
    for k in range(nbImages) :
        Vfinal.extend(copy.deepcopy(V))
    return Vfinal    




def find_4_corners(point,V,width,height,shift):
    coords = [0,0,0,0]

    widthAux = float(width-1)/(widthStep)
    heightAux = float(height-1)/(heightStep)
    # print("WIDTH",widthAux,heightAux)
    x = int(np.floor(point[0]/widthAux))
    y = int(np.floor(point[1]/heightAux))
    # print("X,Y",x,y)
    ind = 2 * (y *(widthStep+1) + x)

        # print("Vy",V[ind+2],"y",point[1],"Vx",V[ind+3],"x",point[0])
    assert(V[ind]<=point[1]and V[ind+1]<=point[0])
    assert(V[ind+2]<=point[1]and V[ind+3]>=point[0])
    assert(V[ind+2*(widthStep+1)]>=point[1]and V[ind+2*(widthStep+1)+1]<=point[0])
    assert(V[ind+2+2*(widthStep+1)]>=point[1]and V[ind+3+2*(widthStep+1)]>=point[0])
   

    return [ind,ind+2,ind+2*(widthStep+1),ind+2+2*(widthStep+1)]
    


def create_w(corner,point,V,warp):
    # Essayer avec des sparses
    W = np.zeros((2,len(V)))
    # print("LE V",V)
    w1 = (V[corner[3]]-point[0])*(V[corner[3]+1]-point[1])
    w2 = (point[0]-V[corner[2]])*(V[corner[2]+1]-point[1])
    w3 = (point[0]-V[corner[0]])*(point[1]-V[corner[0]+1])
    w4 = (V[corner[1]]-point[0])*(point[1]-V[corner[1]+1])
    # print("corner",corner)
    # print("LES W",w1,w2,w3,w4)

    W[0,warp+corner[0]] = w1
    W[1,warp+corner[0]+1] = w1  

    W[0,warp+corner[1]] = w2
    W[1,warp+corner[1]+1] = w2 

    W[0,warp+corner[2]] = w3
    W[1,warp+corner[2]+1] = w3 

    W[0,warp+corner[3]] = w4
    W[1,warp+corner[3]+1] = w4 

    return W

    




def create_weight_a(images,V,dico,points):
    """ Energy 1 """


    size = len(V)/len(images)
    nbImages,height,width = np.shape(images)[:3]
    weights = []
    weightsTranspose = []
    dicAux = {}
    compteur =0
    nbN = []
    for key in dico.keys():
        indexList = dico[key]
        print(np.array(indexList))
        for indexPoint1,indexPoint2 in np.array(indexList).astype(int) :
    
            corner1 = find_4_corners(points[key[0]][indexPoint1],V,width,height,key[0]*size)
            corner2 = find_4_corners(points[key[1]][indexPoint2],V,width,height,key[1]*size)

            W1 = create_w(np.array(corner1),points[key[0]][indexPoint1],V,key[0]*size)
            W2 = create_w(np.array(corner2),points[key[1]][indexPoint2],V,key[1]*size)



            Waux = csr_matrix(W1-W2)

            Waux2 = csc_matrix(np.transpose(W1-W2))
            weights.append(copy.deepcopy(Waux))
            weightsTranspose.append(copy.deepcopy(Waux2))
            
            # Mise a jour du poid dans la cellule :
            if corner1[0] in dicAux.keys():
                dicAux[corner1[0]].append(compteur)
            else :
                dicAux[corner1[0]] = [compteur]
            compteur+=1 
        nbN = np.zeros(len(weights))
        for key in dicAux.keys():
            for index in dicAux[key]:
                nbN[index] = len(dicAux[key])


    return weights,weightsTranspose,nbN



def create_weight_r(images,V):
    """ for 2nd energy : regularisation """
    size = len(V)/len(images)
    nbImages,height,width = np.shape(images)[:3]
    weights = []
    weightsTranspose = []
    for index in range(0,len(V),2):

        W  = np.zeros((2,len(V)))
        

        # Point courant
        W[0][index] = 1.
        W[1][index+1] = 1.


        imageNumber = index/size
        row = (index- imageNumber *size) /(2*(widthStep+1))
        column = ((index - imageNumber *size) % (2*(widthStep+1)))/2


        horizontalLeftBorder = (column == 0)
        horizontalRightBorder = (column >=widthStep)
        verticalUpBorder = (row == 0)
        verticalDownBorder = (row >= heightStep)


        if len(np.where([horizontalLeftBorder,horizontalRightBorder,verticalDownBorder,verticalUpBorder])[0])>=2 :
            print("Corner at : ")    
            print(imageNumber,row,column)
            W[0][index] = 1
            W[1][index] = 1
            continue
            # if horizontalLeftBorder :
            #     W[1][index+3] = -1
            # if horizontalRightBorder :
            #     W[1][index-1] = -1
            # if verticalUpBorder :
            #     W[0][index+2*(widthStep+1)] = -1
            # if verticalDownBorder :
            #     W[0][index-2*(widthStep+1)] = -1
            

            # Waux = csr_matrix(W)
            # Waux2 = csr_matrix(np.transpose(W))
            # weights.append(copy.deepcopy(Waux))
            # weightsTranspose.append(copy.deepcopy(Waux2))



        else :
            W[0][index] = 1.
            W[1][index+1] = 1.
            if horizontalLeftBorder or horizontalRightBorder :
                W[0][index+2*(widthStep+1)] = -1/2.
                W[1][index+2*(widthStep+1)+1] = -1/2.
                W[0][index-2*(widthStep+1)] = -1/2.
                W[1][index-2*(widthStep+1)+1] = -1/2.

            elif verticalUpBorder or verticalDownBorder :
                W[0][index+2] = -1/2.
                W[1][index+3] =  -1/2.
                W[0][index-2] = -1/2.
                W[1][index-1] =  -1/2.

            else :
                W[0][index+2] = -1/4.
                W[1][index+3] =  -1/4.
                W[0][index-2] = -1/4.
                W[1][index-1] =  -1/4.
                W[0][index+2*(widthStep+1)] = -1/4.
                W[1][index+2*(widthStep+1)+1] = -1/4.
                W[0][index-2*(widthStep+1)] = -1/4.
                W[1][index-2*(widthStep+1)+1] = -1/4.



            Waux = csr_matrix(W)
            Waux2 = csr_matrix(np.transpose(W))
            weights.append(copy.deepcopy(Waux))
            weightsTranspose.append(copy.deepcopy(Waux2))

    # return weights
    return weights,weightsTranspose


def create_w_corner(V,images):
    """ For 3rd energy """
    Wcorner = []
    size = len(V)/len(images)
    nbImages,height,width = np.shape(images)[:3]
    shift = 0
    for image in range(len(images)):
        Wtl = np.zeros((2,len(V)))
        Wtr = np.zeros((2,len(V)))
        Wbl = np.zeros((2,len(V)))
        Wbr = np.zeros((2,len(V)))

        Wtl[0][shift] = 1
        Wtl[1][shift+1] = 1


        Wtr[0][shift+2*(widthStep)] = 1
        Wtr[1][shift+2*(widthStep)+1] = 1


        Wbl[0][shift+size-2*widthStep-2] = 1
        Wbl[1][shift+size-2*widthStep-1] = 1


        Wbr[0][shift + size -2] = 1
        Wbr[1][shift + size -1] = 1


        shift +=size

        Wcorner.append(copy.deepcopy(np.array([Wtl,Wtr,Wbl,Wbr])))
    return Wcorner


def pre_simple_b(Wcorner):
    """ ORDRE ??????"""
    B = []
    for imageCorner in Wcorner :
        Bt = (imageCorner[1]-imageCorner[0])
        Bb = (imageCorner[3]-imageCorner[2])
        Br = (imageCorner[3]-imageCorner[1])
        Bl = (imageCorner[2]-imageCorner[0])

        B.append(copy.deepcopy(np.array([Bt,Bb,Bl,Br])))
    return B

def simple_b(Wcorner,V):
    
    Btot = pre_simple_b(Wcorner)
    B = []
    for Bcorner in Btot :
        Bt = np.matmul(Bcorner[0],V)
        Bb = np.matmul(Bcorner[1],V)
        Br = np.matmul(Bcorner[2],V)
        Bl = np.matmul(Bcorner[3],V)

        B.append(copy.deepcopy(np.array([Bt,Bb,Bl,Br])))
    return B


def normalizedB(Wcorner,V):
    """ Non utilise """
    Btot = pre_simple_b(Wcorner)
    B = []
    for Bcorner in Btot :
        Bt = np.matmul(Bcorner[0],V)
        Bb = np.matmul(Bcorner[1],V)
        Br = np.matmul(Bcorner[2],V)
        Bl = np.matmul(Bcorner[3],V)


        Bt = Bt/np.linalg.norm(Bt)
        Bb = Bb/np.linalg.norm(Bb)
        Br = Br/np.linalg.norm(Br)
        Bl = Bl/np.linalg.norm(Bl)
    
        B.append(copy.deepcopy(np.array([Bt,Bb,Bl,Br])))
    
    return B


def orthogonalB(B):
    """ Non utilise """
    Bortho = []
    for b in B :
        Baux = []
        for vect in b :
            
            if abs(vect[0])>1e-1 :
                print(vect[0])
                baux = [-vect[1]/vect[0],1]
            else :
                baux = [1,-vect[0]/vect[1]]
            print(vect,baux)
            Baux.append(baux/np.linalg.norm(baux))
        
        # borthoAux = copy.deepcopy(spectral.orthogonalize(b[0:3:2]))
        Bortho.append(copy.deepcopy(Baux))
    # for corner in B :
    #     Q,R = np.linalg.qr(corner)
    #     Bortho.append(Q)
 
    
    return Bortho



def calculate_scaling_factor(images,dico,points):
    dicScale = {}
    for key in dico.keys():
        indexList = dico[key]
        ptsA = []
        ptsB = []
        for indexPoint1,indexPoint2 in np.array(indexList).astype(int) :
            ptsA.append(points[key[0]][indexPoint1])
            ptsB.append(points[key[1]][indexPoint2])

        hullA = ConvexHull(ptsA)
        verticesA = hullA.vertices.tolist() + [hullA.vertices[0]]
        hullB = ConvexHull(ptsB)
        verticesB = hullB.vertices.tolist() + [hullB.vertices[0]]

        perimeterA = 0
        perimeterB = 0
        for j in range(len(verticesA)-1):
            perimeterA+=np.linalg.norm(ptsA[verticesA[j]]-ptsA[verticesA[j+1]])
            perimeterB+=np.linalg.norm(ptsB[verticesA[j]]-ptsB[verticesA[j+1]])


        dicScale[key] = float(perimeterA)/perimeterB
    return dicScale

def optim_scale(images,dico,points):
    global dicScale
    dicScale = calculate_scaling_factor(images,dico,points)
    x0 = np.ones((len(images)),dtype = float)


    def func(x0):
        sum = 0
        for key in dicScale.keys():
            sum += (dicScale[key] * x0[key[1]]-x0[key[0]]) ** 2
        return sum

    

    def jacobian_func(x):
        jacobian = np.zeros((len(images)))
        for key in dico.keys():
                jacobian[key[0]] += -2*(dicScale[key]*x[key[1]]-x[key[0]])
                jacobian[key[1]] += 2*dicScale[key]*(dicScale[key]*x[key[1]]-x[key[0]])
        return jacobian

    
    eqConstraints = {
    'type': 'eq',
    'fun' : lambda x : np.sum(x0)-len(images),
    'jac' : lambda x : np.ones((len(images)),dtype = float)
    }

    res = minimize(func, x0, method='SLSQP', jac=jacobian_func, constraints=[eqConstraints], options={'tol': 1e-9, 'disp': True})
    print("results",res.x)
    return res.x





def calculate_shape(Wcorner,V):
    listSi = []
    for imageCorner in Wcorner :
        Bt = np.matmul(imageCorner[0]-imageCorner[1],V)
        Bb = np.matmul(imageCorner[2]-imageCorner[3],V)
        Br = np.matmul(imageCorner[1]-imageCorner[3],V)
        Bl = np.matmul(imageCorner[0]-imageCorner[2],V)

        SI = [np.linalg.norm(Bt)+np.linalg.norm(Bb), np.linalg.norm(Bl)+np.linalg.norm(Br)]
        listSi.append(SI)

    return np.array(listSi)


def optimisation_mesh_v2(images,dico,points) :

    V = create_grid(images)
    V2 = create_grid(images)
    dicoAux = copy.deepcopy(dico) # Calcul des shapes demandent bcp de points
    for key in dicoAux.keys():
        if len(dicoAux[key])<=10 :
            del dicoAux[key]
            continue

    Wa,WaTransposed,nbN = create_weight_a(images,V,dico,points) # Tableau de W_i - W_j
    print("Wa",Wa[0].toarray())
    Wr,WrTransposed = create_weight_r(images,V) # Tableau pour tout v de Wv - 1/Nv * Sum(Wv_i)
  
    s = optim_scale(images,dicoAux,points)
    
    Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
    Sinit = copy.deepcopy(calculate_shape(Wcorner,V))

    def energyA(V) :
        # Calcul A :
        A = []
        for w in Wa :
            A.append(np.linalg.norm(w.dot(V))**2)
        resultA = np.sum(np.dot(1/nbN,A))
        return resultA

    def energyB(V):
        # Calcul B :
        B = []
        for w in Wr :
            print(w)
            B.append(np.linalg.norm(w.dot(V))**2)
        resultB = lambdaB *np.sum(B)
        # resultB = 0
        return resultB
    
    def energyC(V):
        #Calcul C :
        Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
        Scurrent = copy.deepcopy(calculate_shape(Wcorner,V))
        resultC =0
        for i in range(len(Scurrent)):
            resultC = lambdaC * np.linalg.norm(Scurrent[i]-s[i]*Sinit[i])**2
        return resultC

    def approximateEnergyC(V):
        WcornerAux = create_w_corner(V,images)
        Bnorm = normalizedB(WcornerAux,V)
        Bortho = orthogonalB(Bnorm)
        Bsimple = simple_b(Wcorner,V)
        

        Es1=0
        
        for i in range(len(Bnorm)) :
            Es1+=(np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0])+np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]) - s[i]*Sinit[i][0])**2
            Es1+=(np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2])+np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]) - s[i]*Sinit[i][1])**2
            # print(np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0]),np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]),Sinit[i][0])
            # print(Bnorm[i][2],Bsimple[i][0],np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][0]),np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]),Sinit[i][1])
        
        Es2 = 0
        for i in range(len(Bortho)):
            Es2 += (np.matmul(np.transpose(Bortho[i][0]),Bsimple[i][0])**2 \
                + np.matmul(np.transpose(Bortho[i][1]),Bsimple[i][1])**2 \
                + np.matmul(np.transpose(Bortho[i][2]),Bsimple[i][2])**2 \
                + np.matmul(np.transpose(Bortho[i][3]),Bsimple[i][3])**2)
        
        # Es2 = 0
        # for i in range(len(Bortho)):

        #     Es2 += (np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0])**2 \
        #         + np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1])**2 \
        #         + np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2])**2 \
        #         + np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3])**2)
            
        
        return lambdaC*(Es1+lambdas*Es2)
        # return lambdaC* Es1

    def energy(V):
        resultA = energyA(V)
        resultB = energyB(V)
        resultC = energyC(V)
        print("Energy A {}".format(resultA))
        print("Energy B {}".format(resultB))
        print("Energy C {}".format(resultC))
        print("Energy Total {}".format(resultA+resultB+resultC))
    
        return resultA+resultB+resultC

    def calculate_matrix_left(V):

        # Calculate A term :
        Aa = np.zeros(np.shape(WaTransposed[0].dot(Wa[0])))
        for i in tqdm.tqdm(range(len(WaTransposed))):
            
           Aa+=(1/nbN[i]*(WaTransposed[i].dot(Wa[i])))
        print("SHAPE Aa",np.shape(Aa))

        

        # Calcul R term :
        Ar = np.zeros(np.shape(WrTransposed[i].dot(Wr[i])))
        for i in range(len(Wr)):
            Ar+=(lambdaB * WrTransposed[i].dot(Wr[i]))
        print("Shape Ar", np.shape(Ar))



        # Calculate S term :
   
        WcornerAux = create_w_corner(V,images)
        Bnorm = normalizedB(WcornerAux,V)
        Bortho = orthogonalB(Bnorm)
        Bsimple = simple_b(Wcorner,V)
        BpreSimple = pre_simple_b(Wcorner)
        
        print(np.shape(WcornerAux))
        As1 = np.zeros(np.shape(np.transpose(V)))
        for i in range(len(Bnorm)):
            As1+=(np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0])+np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]) - 2*s[i]*Sinit[i][0])\
                 * (np.matmul(np.transpose(Bnorm[i][0]),BpreSimple[i][0])+np.matmul(np.transpose(Bnorm[i][1]),BpreSimple[i][1]))
            As1+=(np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2])+np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]) - 2*s[i]*Sinit[i][1])\
                 * (np.matmul(np.transpose(Bnorm[i][2]),BpreSimple[i][2])+np.matmul(np.transpose(Bnorm[i][3]),BpreSimple[i][3]))
        print("As1",np.shape(As1))


        As2 = np.zeros(np.shape(np.matmul(np.transpose(Bnorm[i][0]),WcornerAux[i][0]-WcornerAux[i][1])))
        for i in range(len(Bortho)):
            As2 += np.matmul(np.transpose(Bortho[i][0]),Bsimple[i][0]) * np.matmul(np.transpose(Bortho[i][0]),BpreSimple[i][0])\
                + np.matmul(np.transpose(Bortho[i][1]),Bsimple[i][1]) * np.matmul(np.transpose(Bortho[i][1]),BpreSimple[i][1])  \
                + np.matmul(np.transpose(Bortho[i][2]),Bsimple[i][2]) * np.matmul(np.transpose(Bortho[i][2]),BpreSimple[i][2]) \
                + np.matmul(np.transpose(Bortho[i][3]),Bsimple[i][3]) * np.matmul(np.transpose(Bortho[i][3]),BpreSimple[i][3])
            
        print("As2",np.shape(As2))


        # As2 = np.zeros(np.shape(np.matmul(np.transpose(Bnorm[i][0]),WcornerAux[i][0]-WcornerAux[i][1])))
        # for i in range(len(Bortho)):
        #     As2 += np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0]) * np.matmul(np.transpose(Bnorm[i][0]),BpreSimple[i][0])\
        #         + np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]) * np.matmul(np.transpose(Bnorm[i][1]),BpreSimple[i][1])  \
        #         + np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2]) * np.matmul(np.transpose(Bnorm[i][2]),BpreSimple[i][2]) \
        #         + np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]) * np.matmul(np.transpose(Bnorm[i][3]),BpreSimple[i][3])
            
        # print("As2",np.shape(As2))

        return Aa,Ar,As1,As2
    
    print("ecart entre energy",energyC(V),approximateEnergyC(V),energyC(V)-approximateEnergyC(V))
    print("energy total",energy(V))
    # calculate_matrixLeft(V)

    def optimAux(V):
        for i in range(5):
            
            E = energy(V)
            print(E)
            Ec = energyC(V)
            EappC = approximateEnergyC(V)
            ecart = Ec - EappC
            print("CALCULATE MATRIX LEFT")
            Aa,Ar,As1,As2 = calculate_matrix_left(V)
            print("Aa",Aa)
            print("Ar",Ar)
            print("As1",As1)
            print("As2",As2)
            # As = As1
            As = As1+lambdas*As2
            # left = (np.matmul(np.transpose(Aa),Aa)+np.matmul(np.transpose(Ar),Ar)+np.matmul(np.transpose(As),As))
            left = (np.matmul(np.transpose(Ar),Ar)+np.matmul(np.transpose(As),As))
            right = np.transpose(copy.deepcopy(As)*ecart)
            # print(np.linalg.eig(left))
            # print(np.shape(np.transpose(right)),np.shape(left))
            # V = np.linalg.tensorsolve(left,right)
            print(np.shape(V))
            # V = np.matmul(np.linalg.inv(left),right)
            
            cholesky,lower = scipy.linalg.cho_factor(left,lower = True,check_finite =True)
            V = scipy.linalg.solve(cholesky,right)
            # L = np.linalg.cholesky(left)
            V = np.array(V).reshape(-1)
            print(V)
            newImages = visuGrid(V,images)

            

    optimAux(V)
    










def optimisation_mesh(images,dico,points):

    V = create_grid(images)
    
    V2 = create_grid(images)
    dicoAux = copy.deepcopy(dico) # Calcul des shapes demandent bcp de points
    for key in dicoAux.keys():
        if len(dicoAux[key])<=10 :
            del dicoAux[key]
            continue

    Wa,WaTransposed,nbN = create_weight_a(images,V,dico,points) # Tableau de W_i - W_j
    Wr,WrTransposed = create_weight_r(images,V) # Tableau pour tout v de Wv - 1/Nv * Sum(Wv_i)
    s = optim_scale(images,dicoAux,points)
    print("INIT S",s)
    
    Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages


    Sinit = copy.deepcopy(calculate_shape(Wcorner,V))
    # print("Sinit",Sinit)
    print("V",np.shape(V))
    
    # print("NB PAR CASE ",nbN)

    import time

    def energy(V) :
 
        # Calcul A :
        A = []
        for w in Wa :
            A.append(np.linalg.norm(w.dot(V))**2)
        resultA = np.sum(np.dot(1/nbN,A))

        # Calcul B :
        B = []
        for w in Wr :
            B.append(np.linalg.norm(w.dot(V))**2)
        resultB = lambdaB *np.sum(B)
        # resultB = 0
    

        #Calcul C :
        Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
        Scurrent = copy.deepcopy(calculate_shape(Wcorner,V))
        resultC =0
        for i in range(len(Scurrent)):
            resultC = lambdaC * np.linalg.norm(Scurrent[i]-s[i]*Sinit[i])**2
        

        print("Energy A {}".format(resultA))
        print("Energy B {}".format(resultB))
        print("Energy C {}".format(resultC))
        print("Energy Total {}".format(resultA+resultB+resultC))
        return resultA+resultB+resultC




    def gradEnergyExact(V):
        #Calcul Grad A :
        auxA = []
        for i in range(len(Wa)):
            auxA.append(Wa[i].dot(V))
        auxauxA = []
        for i in range(len(WaTransposed)):
            auxauxA.append(WaTransposed[i].dot(auxA[i]))
        A = np.dot(2/nbN,auxauxA)
 
        print("Gradient A", np.linalg.norm(A))
        print(np.shape(A))

        # Calcul Grad B :
        auxB = []
        for i in range(len(Wr)):
            auxB.append(Wr[i].dot(V))

        
        auxauxB = []
        for i in range(len(Wr)):
            auxauxB.append(WrTransposed[i].dot(auxB[i]))

        
        B = 2 * lambdaB * np.sum(auxauxB,axis=0)
        print("GRADIENT B",np.linalg.norm(B))
        print(np.shape(B))
        # B = np.zeros(np.shape(V))
      

        # Calcul Grad C :
        Scurrent = calculate_shape(Wcorner,V)
        C = np.zeros(np.shape(B))
        Btot = pre_simple_b(Wcorner)
        simpleB = simple_b(Wcorner,V)
        for i in range(len(Scurrent)):
            constant = 2 * Scurrent[i]-s[i]*Sinit[i]
            
            # Up = Wcorner[i][0]-Wcorner[i][1]
            # Down = Wcorner[i][2]-Wcorner[i][3]
            # Left = Wcorner[i][0]-Wcorner[i][2]
            # Right = Wcorner[i][1]-Wcorner[i][3]
            #MAYBE OPPOSITE WAY FOR UP AND DOWN
            #=====================================================
            # C+= constant[0]*(np.matmul(np.transpose(),np.matmul(Up,V))+np.matmul(np.transpose(Down),np.matmul(Down,V)))
            C+= constant[0]*(np.matmul(np.transpose(Btot[i][0]),simpleB[i][0])+np.matmul(np.transpose(Btot[i][1]),simpleB[i][1]))
            # C+= constant[1]*(np.matmul(np.transpose(Left),np.matmul(Left,V))+np.matmul(np.transpose(Right),np.matmul(Right,V)))
            C+= constant[1]*(np.matmul(np.transpose(Btot[i][2]),simpleB[i][2])+np.matmul(np.transpose(Btot[i][3]),simpleB[i][3]))
        C = lambdaC * C
        
        print("C Grad VALUE",np.linalg.norm(C))
        print(np.shape(C))
        print("GRADFINAL",np.shape(A+B+C))
        return B+C


    def gradEnergyApproximated(V):
            #Calcul Grad A :
        auxA = []
        for i in range(len(Wa)):
            auxA.append(Wa[i].dot(V))
        auxauxA = []
        for i in range(len(WaTransposed)):
            auxauxA.append(WaTransposed[i].dot(auxA[i]))
        A = np.dot(2/nbN,auxauxA)
 
        print("Gradient A", np.linalg.norm(A))
        print(np.shape(A))

        # Calcul Grad B :
        auxB = []
        for i in range(len(Wr)):
            auxB.append(Wr[i].dot(V))

        
        auxauxB = []
        for i in range(len(Wr)):
            auxauxB.append(WrTransposed[i].dot(auxB[i]))

        
        B = 2 * lambdaB * np.sum(auxauxB,axis=0)
        print("GRADIENT B",np.linalg.norm(B))
        print(np.shape(B))
        # B = np.zeros(np.shape(V))
      

        # Calcul Grad C :

        WcornerAux = create_w_corner(V,images)
        Bnorm = normalizedB(WcornerAux,V)
        Bortho = orthogonalB(Bnorm)
        Bsimple = simple_b(Wcorner,V)
        BpreSimple = pre_simple_b(Wcorner)
      
        As1 = np.zeros(np.shape(np.transpose(V)))
        for i in range(len(Bnorm)):
            As1+=(np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0])+np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]) - 2*s[i]*Sinit[i][0])\
                 * (np.matmul(np.transpose(Bnorm[i][0]),BpreSimple[i][0])+np.matmul(np.transpose(Bnorm[i][1]),BpreSimple[i][1]))
            As1+=(np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2])+np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]) - 2*s[i]*Sinit[i][1])\
                 * (np.matmul(np.transpose(Bnorm[i][2]),BpreSimple[i][2])+np.matmul(np.transpose(Bnorm[i][3]),BpreSimple[i][3]))
        # print("As1",np.shape(As1))
        

        As2 = np.zeros(np.shape(np.matmul(np.transpose(Bnorm[i][0]),WcornerAux[i][0]-WcornerAux[i][1])))
        for i in range(len(Bortho)):
            As2 += np.matmul(np.transpose(Bnorm[i][0]),Bsimple[i][0]) * np.matmul(np.transpose(Bnorm[i][0]),BpreSimple[i][0])\
                + np.matmul(np.transpose(Bnorm[i][1]),Bsimple[i][1]) * np.matmul(np.transpose(Bnorm[i][1]),BpreSimple[i][1])  \
                + np.matmul(np.transpose(Bnorm[i][2]),Bsimple[i][2]) * np.matmul(np.transpose(Bnorm[i][2]),BpreSimple[i][2]) \
                + np.matmul(np.transpose(Bnorm[i][3]),Bsimple[i][3]) * np.matmul(np.transpose(Bnorm[i][3]),BpreSimple[i][3])
            
        # print("As2",np.shape(As2))
        C = np.matmul(As1+lambdas*As2,V)
        print("C Grad VALUE",np.linalg.norm(C))
        print(np.shape(C))
        print("GRADFINAL",np.shape(A+B+C))
        return A+B+C

    res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy,V,fprime=gradEnergyApproximated))
    # res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy,V,approx_grad = True,bounds=[(0,None)]*len(V),maxiter = 1))
    # for i in range(10):
        # reconstruction(res[0],images)
        # res = copy.deepcopy(scipy.optimize.fmin_l_bfgs_b(energy,res[0],fprime=gradEnergy,bounds=[(0,None)]*len(V),maxiter = 100))
        
    energy(V2)
    return res





#========#========#========#========#========#========#========#========#========#========#========#========
#RECONSTRUCTION
#========#========#========#========#========#========#========#========#========#========#========#========
import matplotlib.pyplot as plt

def visuGrid(newGrid,images):
    V = create_grid(images)
    size = len(newGrid)/len(images)
    ymax = np.max(newGrid[0:size:2])
    xmax = np.max(newGrid[1:size:2])
    print(xmax,ymax)
    for i in range(len(images)) :
    # for i in range(1) :
        grid = newGrid[i*size:(i+1)*size]
        
        xmap = grid[1::2]
        ymap = grid[0::2]
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

def reconstructionV2(newGrid,images):
    V = np.array(create_grid(images))
    size = len(newGrid)/len(images)
    ymax = np.max(newGrid[0:size:2])
    xmax = np.max(newGrid[1:size:2])

    V = V[:size].reshape(-1,2)
    V = np.transpose(V)
    V = np.transpose([V[1],V[0]])


    previousCompteur = 0
    print(np.shape(V))
    newImages = []
    for i in range(len(images)) :
        print(i)
        grid = newGrid[i*size:(i+1)*size]
        grid = grid.reshape(-1,2)
        grid = np.transpose(grid)
        grid = np.transpose([grid[1],grid[0]])
       
        # print(np.shape(grid))
        image = images[i]
        rows, cols = image.shape[0], image.shape[1]


        tform = PiecewiseAffineTransform()
        tform.estimate(V, grid)
        for k in range(len(V)):
            print(V[k],grid[k])

        out = warp(image, tform,output_shape = (rows+300,cols+300))
        print(i,"out",out)
        fig, ax = plt.subplots()
        ax.imshow(out)
        ax.plot(tform.inverse(V)[:, 0], tform.inverse(V)[:, 1], '.b')
        plt.show()
        newImages.append(out)
    return newImages
# fig, ax = plt.subplots()
# ax.imshow(out)
# ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
# ax.axis((0, out_cols, out_rows, 0))
# plt.show()

def findBestUpperRightCorner(newGrid,x,y,previousIndex = 0):
    index = previousIndex
    compteur = 0
    indexAux = previousIndex
    
    while compteur<2*widthStep  and index<len(newGrid):
        indexAux+=1
        if np.linalg.norm([newGrid[2*indexAux]-y,newGrid[2*indexAux+1]-x])<np.linalg.norm([newGrid[2*index]-y,newGrid[2*index+1]-x]):
            index = indexAux
            compteur =0
        else :
            compteur+=1
  

    
    row = (index) /(2*(widthStep+1))
    column = ((index) % (2*(widthStep+1)))/2


    horizontalLeftBorder = (column == 0)
    horizontalRightBorder = (column >=widthStep)
    verticalUpBorder = (row == 0)
    verticalDownBorder = (row >= heightStep)
    
    # print(index,x,y,newGrid[2*compteur+1],newGrid[2*compteur])
    return index,(horizontalLeftBorder or horizontalRightBorder or verticalDownBorder or verticalUpBorder)

def reconstruction(newGrid,images):
    V = create_grid(images)
    size = len(newGrid)/len(images)
    ymax = np.max(newGrid[0:size:2])
    xmax = np.max(newGrid[1:size:2])
    previousCompteur = 0
    print(xmax,ymax)
    newImages = []
    for i in range(len(images)) :
        grid = newGrid[i*size:(i+1)*size]
        xmap = grid[1::2]
        ymap = grid[0::2]
        yminLoc = np.min(ymap)
        xminLoc = np.min(xmap)
        ymaxLoc = np.max(ymap)
        xmaxLoc = np.max(xmap)
        newMap = np.zeros((int(max(ymaxLoc,len(images[0]))),int(max(xmaxLoc,len(images[1]))),2))
        for y in tqdm.tqdm(range(int(max(ymaxLoc,len(images[0]))))):
            for x in range(int(max(xmaxLoc,len(images[1])))):
                compteur,status = findBestUpperRightCorner(newGrid,x,y,max(previousCompteur - 10,0))
                previousCompteur = compteur
                # print(compteur,status)
                if not status :
                # print(compteur)
                    alpha = (grid[2*compteur+1]-x)/(grid[2*compteur+1]-grid[2*(compteur-1)+1])
                    beta = (grid[2*compteur]-y)/(grid[2*compteur]-grid[2*(compteur-widthStep-1)])
                    
                    newMap[y,x] = [
                                (beta)*V[2*compteur]+(1-beta)*V[2*compteur-2*(heightStep+1)],
                                alpha*V[2*compteur+1]+(1-alpha)*V[2*compteur-1]
                                ]
                else :
                    newMap[y,x] = [
                            V[2*compteur],
                            V[2*compteur+1]
                                ]

        newImage = cv2.remap(images[i],newMap)
        newImages.append(copy.deepcopy(newImage))
    return newImages
        
 #========#========#========#========#========#========#========#========#========#========#========#========
 #MAIN
 #========#========#========#========#========#========#========#========#========#========#========#========



if __name__ == "__main__":
  
    # inputPath = "SimpleWideBaseline{}".format(datasetNumber)
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
        # images[-1]= putNone(images[-1])
    images = np.array(images)
    # print(np.shape(images))
    dico,points = getCommonFeatures(images)
    # print("DICO",dico)
    # import cPickle as pickle
    # with open("dico{}.pck".format(datasetNumber),"wb") as f :
        # pickle.dump(dico,f)

    # with open("dico{}.pck".format(datasetNumber),"rb") as f :
        # dico = pickle.load(f)


    # dico = dico1
    # points = pts1
    # images = np.zeros((2,9,9))
    # V = create_grid(images)
    # Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
    
    # print(np.where(Wcorner[0]))
    # Scurrent = copy.deepcopy(calculate_shape(Wcorner,V))
    # s = [1,1]
    # resultC = 0
    # for i in range(len(images)):
    #     resultC += lambdaC * np.linalg.norm(Scurrent[i]-Scurrent[i]*s[i])**2
    # print(resultC)
        

    # dico = dico2
    # points = pts2
    # images = np.zeros((2,2000,1000))


    # with open("points{}.pck".format(datasetNumber),"wb") as f :
        # pickle.dump(points,f)

    # with open("points{}.pck".format(datasetNumber),"rb") as f :
        # points = pickle.load(f)

    # for k in dico[0,1] :
        # print(points[0,1][k[0]],points[0,1][k[1]])
  
    # print("DICO",dico)


    # V = create_grid(images)
    # corner = create_w_corner(V,images)
    # pre_simple = pre_simple_b(corner)
    # Bsimple = simple_b(corner,V)
    # B = normalizedB(corner,V)
    # Bortho = orthogonalB(B)
    # for i in range(len(B)):
    #     print("=========================")
    #     print(i)
    #     print("presimple",pre_simple[i][0])
    #     print("Bsimple",Bsimple[i])
    #     print("Corner",np.where(corner[i]>0))
    #     print("Normalized",B[i])
    #     print("Ortho",Bortho[i])
        # print("=========================")
    # Wa,nbN = create_weight_a(images,V,dico,points)
    # Wr = create_weight_r(images,V)
    # optim_scale(images,dico,points)
    # A = optimisation_mesh_v2(images,dico,points)
    A = optimisation_mesh(images,dico,points)
    # with open("resultSparse{}.pck".format(datasetNumber),"wb") as f :
        # pickle.dump(A,f)


    # with open("resultSparse{}.pck".format(datasetNumber),"rb") as f :
        # A = pickle.load(f)
    # print(A[0])
    visuGrid(A[0],images)
    newImages = reconstructionV2(A[0],images)
    # print(np.shape(newImages))
   