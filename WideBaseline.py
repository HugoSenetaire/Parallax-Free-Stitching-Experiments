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

# Constants :

R = 50
gamma = 10
heightStep = 20
widthStep = 10


# For features detection/description without threading
def detect_and_describe(image, width = 600):
    image = np.array(image)
    # Get Width & Height of the image
    # print(image)
    H, W = image.shape[:2]

    widthFactor = 1.0 * width / image.shape[1]

    # Resize & grayify image
    S = np.array([[widthFactor, 0.0, 0.0], [0.0, widthFactor, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    small_image = cv2.warpPerspective(image, S, (int(W * widthFactor), int(H * widthFactor)))
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

    # Detection keyponts
    surf = cv2.xfeatures2d.SURF_create(nOctaves=2, upright=True)
    kps = surf.detect(gray)
    if not kps:
        surf = cv2.xfeatures2d.SURF_create(upright=True)
        kps = surf.detect(gray)

    # Compute descriptors
    daisy = cv2.xfeatures2d.DAISY_create()
    kps, descriptors = daisy.compute(gray, kps)
    kps = np.float32([kp.pt for kp in kps])

    return kps, descriptors, S




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

def featureMatching(featuresA,featuresB):
    
    
    kpsA, descriptorA, SA = featuresA
    kpsB, descriptorB, SB = featuresB
    # kpsA, descriptorA, SA = detect_and_describe(featuresA)
    # kpsB, descriptorB, SB = detect_and_describe(featuresB)
    matcher = cv2.DescriptorMatcher_create("FlannBased")
    ratio=0.75
    matches = matcher.knnMatch(descriptorA, descriptorB, 2)  
    matches = filter(lambda m: len(m) == 2 and m[0].distance < m[1].distance * ratio, matches)
    matches = [(m[0].trainIdx, m[0].queryIdx) for m in matches]

    ptsA = np.array(np.float32([kpsA[k] for (_, k) in matches]))
    ptsB = np.array(np.float32([kpsB[k] for (k, _) in matches]))

    indexA = np.array(np.float32([k for (_, k) in matches]))
    indexB = np.array(np.float32([k for (k, _) in matches]))


    # Calculate Homography
    listCommonA = outlier_rejection(ptsA,ptsB)
    listCommonB = outlier_rejection(ptsB,ptsA)
    listCommon = np.union1d(listCommonA,listCommonB)

    listeCouple = []
    # listeCouple = np.array([indexA[k],indexB[k]] for k in listCommon)
    for k in listCommon :
        listeCouple.append([indexA[k],indexB[k]])

    return np.array(listeCouple)

def get_features(image):
    kpsA, descriptorA, SA = detect_and_describe(image)
    return [kpsA,descriptorA,SA]


def getCommonFeatures(images):
    dico = {}
    points = {}
    features = {}

    for i in range(len(images)):
        features[i] = get_features(images[i])
        points[i] = features[i][0]

    for i in range(len(images)-1):
        dico[i,i+1] = []
        listCommon = featureMatching(features[i],features[i+1])
        for index in listCommon :
            dico[i,i+1].append([index[0],index[1]])

    return dico,points



def get_perspective_transform_matrix(p1, p2):
    matrixIndex = 0
    A = np.zeros((2*len(p1), 9))
    for i in range(0, len(p1)):
        x = p1[i][1]
        y = p1[i][0]

        u = p2[i][1]
        v = p2[i][0]

        # A[matrixIndex] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
        # A[matrixIndex + 1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

        A[matrixIndex] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        A[matrixIndex + 1] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]

        matrixIndex = matrixIndex + 2

    U, s, V = np.linalg.svd(A, full_matrices=True)
    
    matrix = V[:, 8].reshape(3, 3)
    return matrix



def outlier_rejection(ptsA,ptsB):
    listCommon = []
    for point in tqdm.tqdm(ptsA) :
        listNeighboursA,listNeighboursB,listIndexNeighbours = findNeighboursIndices(point,ptsA,ptsB)
        if len(listNeighboursA)>4 and len(listNeighboursB)>4:
            h, status = cv2.findHomography(np.array(listNeighboursA), np.array(listNeighboursB))
            # h = get_perspective_transform_matrix(listNeighboursA,listNeighboursB)
            for i in range(len(listNeighboursA)):
                if np.linalg.norm(apply_homography(h,listNeighboursA[i])-listNeighboursB[i])**2<gamma :
                    if listIndexNeighbours[i] not in listCommon : 
                        listCommon.append(listIndexNeighbours[i])
               
    return listCommon


def apply_homography(homography,point):
    newPoint = [point[0],point[1],0]
    result = np.dot(homography,newPoint)
    return result[:2]


def create_grid(images,heightStep = heightStep,widthStep = widthStep):
    nbImages,height,width = np.shape(images)[:3]
    # print("SHAPE IMAGE", np.shape(images))

    V = []
    for i in range(heightStep+1) :
        for j in range(widthStep+1):
            V.append(i*float(height)/(heightStep))
            V.append(j*float(width)/(widthStep))
    Vfinal = []
    for k in range(nbImages) :
        Vfinal.extend(copy.deepcopy(V))
    return Vfinal    




def find_4_corners(point,V,width,height,shift):
    coords = [0,0,0,0]
    widthAux = float(width)/widthStep
    # print("widthAux",widthAux)
    heightAux = float(height)/heightStep
    # print("heightAux",heightAux)

    # x = np.floor(point[1]/widthAux)
    # y = np.floor(point[0]/heightAux)
    
    y = np.floor(point[0]/heightAux)
    x = np.floor(point[1]/widthAux)
    # print("point",point[1],point[0])
    # print("x : {} , y : {}".format(x*widthAux,y*heightAux))
    # print("x {} y {}".format(x,y))
    
    # index = 2 * (x *(widthStep+1) + y)
    ind = int(2 * (y *(widthStep+1) + x))
   
    # print(V[ind:ind+30])
    return [ind,ind+2,ind+2*(widthStep+1),ind+2+2*(widthStep+1)]
    


def create_w(corner,point,V,warp):
    # Essayer avec des sparses
    W = np.zeros((2,len(V)))
    # print("create_w",corner,point,V,np.shape(V),np.shape(W))
    # print(corner[2])
    # print(V[corner[2]+1])
    # print(point[1])
    w1 = (V[corner[2]]-point[0])*(V[corner[2]+1]-point[1])
    w2 = (point[0]-V[corner[3]])*(V[corner[3]+1]-point[1])
    w3 = (point[0]-V[corner[0]])*(point[1]-V[corner[0]+1])
    w4 = (V[corner[1]]-point[0])*(point[1]-V[corner[1]+1])


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
    dicAux = {}
    compteur =0
    nbN = []
    for i in range(len(images)-1):
        indexList = dico[i,i+1]
        for indexPoint1,indexPoint2 in np.array(indexList).astype(int) :
        # for i in range(len(indexList)):
            # indexPoint1 = indexList[i][0]
            # indexPoint2 = indexList[i][1]
       
            corner1 = find_4_corners(points[i][indexPoint1],V,width,height,i*size)

   
            # print("corner1",points[i][indexPoint1])
            # print(corner1)
            # print(V[int(corner1[0])],V[int(corner1[0]+1)])
            # print(V[int(corner1[1])],V[int(corner1[1]+1)])
            # print(V[int(corner1[2])],V[int(corner1[2]+1)])
            # print(V[int(corner1[3])],V[int(corner1[3]+1)])
            
            corner2 = find_4_corners(points[i+1][indexPoint2],V,width,height,i*size)
            # print("corner2",points[i+1][indexPoint2],V[int(corner2[0])],V[int(corner2[1])])
            # print(corner2)
            # print(V[int(corner2[0])],V[int(corner2[0]+1)])
            # print(V[int(corner2[1])],V[int(corner2[1]+1)])
            # print(V[int(corner2[2])],V[int(corner2[2]+1)])
            # print(V[int(corner2[3])],V[int(corner2[3]+1)])
            # print(indexPoint1,indexPoint2,i,i+1,np.shape(V),np.shape(points[i]))
            W1 = create_w(np.array(corner1),points[i][indexPoint1],V,i*size)
            W2 = create_w(np.array(corner2),points[i+1][indexPoint2],V,(i+1)*size)

            # print("W1",W1[0])
            # print("W2",W2[0])
            weights.append(W1-W2)
            
            # Mise a jour du poid dans la cellule :
            if corner1[0] in dicAux.keys():
                dicAux[corner1[0]].append(compteur)
            else :
                dicAux[corner1[0]] = [compteur]
            compteur+=1 
    # print(dicAux)
    nbN = np.zeros(len(weights))
    for key in dicAux.keys():
        for index in dicAux[key]:
            nbN[index] = len(dicAux[key])


    return weights,nbN



def create_weight_r(images,V):
    """ for 2nd energy : regularisation """
    size = len(V)/len(images)
    nbImages,height,width = np.shape(images)[:3]
    weights = []
    weightsTranspose = []
    for index in range(0,len(V),2):

        W  = np.zeros((2,len(V)))
        
        W[0][index] = 1.
        W[1][index+1] = 1.
        i = index
        imageNumber = index/size
        row = (index- imageNumber *size) /(2*(widthStep+1))
        column = ((index - imageNumber *size) % (2*(widthStep+1)))/2
        horizontalLeftBorder = (column == 0)
        horizontalRightBorder = (column >=widthStep)
        verticalUpBorder = (row == 0)
        verticalDownBorder = (row >= heightStep)
        # print(imageNumber,row,column)
        # print(len(np.where([horizontalLeftBorder,horizontalRightBorder,verticalDownBorder,verticalUpBorder])[0]))
        if len(np.where([horizontalLeftBorder,horizontalRightBorder,verticalDownBorder,verticalUpBorder])[0])>=2 :
            print("Corner at : ")    
            print(imageNumber,row,column)

        elif horizontalLeftBorder or horizontalRightBorder :
            W[0][i+2*(widthStep+1)] = -1/2.
            W[1][i+2*(widthStep+1)+1] = -1/2.
            W[0][i-2*(widthStep+1)] = -1/2.
            W[1][i-2*(widthStep+1)+1] = -1/2.

        elif verticalUpBorder or verticalDownBorder :
            W[0][i+2] = -1/2.
            W[1][i+3] =  -1/2.
            W[0][i-2] = -1/2.
            W[1][i-1] =  -1/2.

        else :
            W[0][i+2] = -1/4.
            W[1][i+3] =  -1/4.
            W[0][i-2] = -1/4.
            W[1][i-1] =  -1/4.
            W[0][i+2*(widthStep+1)] = -1/4.
            W[1][i+2*(widthStep+1)+1] = -1/4.
            W[0][i-2*(widthStep+1)] = -1/4.
            W[1][i-2*(widthStep+1)+1] = -1/4.


        Waux = csr_matrix(W)
        Waux2 = csc_matrix(np.transpose(W))
        # weights.append(copy.deepcopy(W))
        weights.append(copy.deepcopy(Waux))
        weightsTranspose.append(copy.deepcopy(Waux2))

    # return weights
    return weights,weightsTranspose


def create_w_corner(V,images):
    """ For 3rd energy """
    Wcorner = []
    size = len(V)/len(images)
    # print(size,widthStep,heightStep)
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


        Wbl[0][shift+size-2*widthStep -2] = 1
        Wbl[1][shift+size - 2*widthStep-1] = 1


        Wbr[0][shift + size -2] = 1
        Wbr[1][shift + size -1] = 1


        shift +=size

        Wcorner.append(copy.deepcopy(np.array([Wtl,Wtr,Wbl,Wbr])))
    return Wcorner

def normalizedB(Wcorner,V):
    B = []
    for imageCorner in Wcorner :
        Bt = np.matmul(imageCorner[0]-imageCorner[1],V)
        Bb = np.matmul(imageCorner[2]-imageCorner[3],V)
        Br = np.matmul(imageCorner[1]-imageCorner[3],V)
        Bl = np.matmul(imageCorner[0]-imageCorner[2],V)

        Bt = Bt/np.linalg.norm(Bt)
        Bb = Bt/np.linalg.norm(Bb)
        Br = Bt/np.linalg.norm(Br)
        Bl = Bt/np.linalg.norm(Bl)
    
    B.append(copy.deepcopy(np.array([Bt,Bb,Br,Bl])))
    return B


def orthogonalB(B):
    """ A TESTER """
    Bortho = []
    for corner in B :
        Q,R = np.linalg.qr(corner)
        Bortho.append(Q)

    return Bortho



def calculate_scaling_factor(images,dico,points):
    dicScale = {}
    for i in range(len(images)-1):
        indexList = dico[i,i+1]
        ptsA = []
        ptsB = []
        for indexPoint1,indexPoint2 in np.array(indexList).astype(int) :
            ptsA.append(points[i][indexPoint1])
            ptsB.append(points[i+1][indexPoint2])

        hullA = ConvexHull(ptsA)
        verticesA = hullA.vertices.tolist() + [hullA.vertices[0]]
        hullB = ConvexHull(ptsB)
        verticesB = hullB.vertices.tolist() + [hullB.vertices[0]]

        perimeterA = 0
        perimeterB = 0
        for j in range(len(verticesA)-1):
            perimeterA+=np.linalg.norm(ptsA[verticesA[j]]-ptsA[verticesA[j+1]])
            perimeterB+=np.linalg.norm(ptsB[verticesA[j]]-ptsB[verticesA[j+1]])


        dicScale[i,i+1] = float(perimeterA)/perimeterB
    return dicScale

def optim_scale(images,dico,points):
    global dicScale
    dicScale = calculate_scaling_factor(images,dico,points)
    # print("dicScale",dicScale)
    x0 = np.ones((len(images)),dtype = float)
    # print("==============+A MODIFIER ================", " Dans optim scale")
    # dicScale[0,1] = 200 # A Supprimer
    def func(x0):
        sum = 0
        for i in range(len(x0)-1):
            sum += (dicScale[i,i+1] * x0[i+1]-x0[i]) ** 2
        # print("sum",sum)
        return sum
    def jacobian_func(x):
        jacobian = np.zeros((len(images)))
        for i in range(len(images)):
            if i <len(images)-1 :
                # print("OK")
                jacobian[i] = -2*(dicScale[i,i+1]*x[i+1]-x[i])
                # print(jacobian)
            if i > 0 :
                # print("ok2")
                jacobian[i] = 2*dicScale[i-1,i]*(dicScale[i-1,i]*x[i]-x[i-1])
                # print(jacobian)
        # print("jacobian",jacobian)
        return jacobian
    eqConstraints = {
    'type': 'eq',
    'fun' : lambda x : np.sum(x0)-len(images),
    'jac' : lambda x : np.ones((len(images)),dtype = float)
    }

    res = minimize(func, x0, method='SLSQP', jac=jacobian_func, constraints=[eqConstraints], options={'tol': 1e-9, 'disp': True})
    # res = minimize(func, x0, method='trust-constr', jac=jacobian_func, constraints=[eqConstraints], options={'disp': True, 'gtol' : 1e-20})
    # print("results",res.x)
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


    


def optimisation_mesh(images,dico,points):
    V = create_grid(images)
    Wa,nbN = create_weight_a(images,V,dico,points) # Tableau de W_i - W_j
    Wr,WrTransposed = create_weight_r(images,V) # Tableau pour tout v de Wv - 1/Nv * Sum(Wv_i)
    s = optim_scale(images,dico,points)
    Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
    Sinit = copy.deepcopy(calculate_shape(Wcorner,V))
    # print("NB PAR CASE ",nbN)

    import time

    def energy(V) :
 
        A = np.linalg.norm(np.dot(Wa,V),axis =1)**2
        # print(np.shape(A))
        # print("WHERE",Wa[0][0])
        # print(A)
        resultA = np.sum(np.dot(1/nbN,A))
  
        # resultA = 0
        # for i in range(len(nbN)):
            # resultA+=(A[i]*1/nbN[i])
        # print(resultA)
        print(np.shape(Wr[0]),np.shape(V))

        # B = np.linalg.norm(np.dot(Wr,V),axis = 1)**2
        B = []
        for w in Wr :
            # print(np.shape(w))
            # print(w.dot(V))
            B.append(np.linalg.norm(w.dot(V)))
        # print(np.shape(B))
        resultB = np.sum(B)
        
        
        # print("result B {}".format(resultB))

        
        Wcorner = create_w_corner(V,images) # W corner (Wtl, Wtr,Wbl,Wbr) * nbImages
        
        Scurrent = calculate_shape(Wcorner,V)
        print("CURRENT - INIT SHAPE",np.shape(Scurrent-s*Sinit))
        resultC = np.sum(np.linalg.norm(Scurrent-s*Sinit,axis =1))
    

        print("Energy A {}".format(resultA))
        print("Energy B {}".format(resultB))
        print("Energy C {}".format(resultC))
        print("Energy Total {}".format(resultA+resultB+resultC))
        return resultA+resultB+resultC




    def gradEnergy(V):
        # print(np.shape(Wa),np.shape(np.transpose(Wa,(0,2,1))))
        # auxA = np.dot(Wa,V)
        auxA = []
        for i in range(len(Wa)):
            auxA.append(np.matmul(np.transpose(Wa[i]),Wa[i]))
        # print(np.shape(auxA))
        # print(np.shape(np.dot(auxA,V)))
        # print(np.shape(np.dot(1/nbN,np.dot(auxA,V))))
        A = np.dot(1/nbN,np.dot(auxA,V))
        # print("A grad",np.shape(A))

        
        # for i in range(len(Wr)):
            # auxB.append(np.matmul(np.transpose(Wr[i]),Wr[i]))
        auxB = []
        for i in range(len(Wr)):
            # print(np.shape(Wr[i].dot(V)))
            auxB.append(Wr[i].dot(V))

        
        auxauxB = []
        for i in range(len(Wr)):
            auxauxB.append(WrTransposed[i].dot(auxB[i]))

        print(np.shape(auxauxB))
        B = np.sum(auxauxB,axis=0)
        # print("B Grad",np.shape(B))
      
        Scurrent = calculate_shape(Wcorner,V)

        # print("Wcorner",Wcorner)
        # print("Scurrent",Scurrent)
        # print("Sinit",Sinit)
        C = np.zeros(np.shape(B))
        for i in range(len(Scurrent)):
            constant = Scurrent[i]-s*Sinit[i]
            # corner = Wcorner[i]

            Up = Wcorner[i][0]-Wcorner[i][1]
            Down = Wcorner[i][2]-Wcorner[i][3]
            Left = Wcorner[i][0]-Wcorner[i][2]
            Right = Wcorner[i][1]-Wcorner[i][3]
            #MAYBE OPPOSITE WAY FOR UP AND DOWN
            #=====================================================


            C+= constant[0]*(np.matmul(np.transpose(Up),np.matmul(Up,V))+np.matmul(np.transpose(Down),np.matmul(Down,V)))
            C+= constant[1]*(np.matmul(np.transpose(Left),np.matmul(Left,V))+np.matmul(np.transpose(Right),np.matmul(Right,V)))
        
        # print("C Grad",np.shape(C))
        # print("C Grad VALUE",C)

        return A+B+C

    res = scipy.optimize.fmin_l_bfgs_b(energy,V,fprime=gradEnergy)
    
    return res



def reconstruction(newGrid,images):
    
    size = len(newGrid)/len(images)
    ymax = np.max(newGrid[0:size:2])
    xmax = np.max(newGrid[1:size:2])
    print(xmax,ymax)
    for i in range(len(images)) :
        grid = newGrid[i*size:(i+1)*size]
        xmap = grid[0:size:2]
        ymap = grid[1:size:2]


        yminLoc = np.max(ymap)
        xminLoc = np.max(xmap)
        ymaxLoc = np.max(ymap)
        xmaxLoc = np.max(xmap)



    
 



if __name__ == "__main__":
    
    inputPath = "SimpleWideBaseline"
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

    # dico,points = getCommonFeatures(images)

    import cPickle as pickle
    # with open("dico.pck","wb") as f :
        # pickle.dump(dico,f)

    with open("dico.pck","rb") as f :
        dico = pickle.load(f)



    # with open("points.pck","wb") as f :
        # pickle.dump(points,f)

    with open("points.pck","rb") as f :
        points = pickle.load(f)

    
    print(len(dico[0,1]))
    # V = create_grid(images)
    # Wa,nbN = create_weight_a(images,V,dico,points)
    # Wr = create_weight_r(images,V)
    # optim_scale(images,dico,points)
    A = optimisation_mesh(images,dico,points)
    with open("resultSparse.pck","wb") as f :
        pickle.dump(A,f)
 