from plyfile import PlyData, PlyElement
import sklearn
import numpy as np
from sklearn.decomposition import PCA



def findPhotoPosition(data,colorPhoto = [0,255,0]):
    photoPos = []
    for point in data : 
        if np.all([point[3],point[4],point[5]]==colorPhoto):
            photoPos.append([point[0],point[1],point[2]])
    return photoPos



def automaticPlaneFinding(data):
    photoPos = findPhotoPosition(data)
    pca = PCA(n_components = 2)
    # print(photoPos)
    pca.fit(photoPos)

    return pca.components_

def addVector(point,vector,color = [255,255,255]):
    x = point[0][0] + vector[0]
    y = point[0][1] + vector[1]
    z = point[0][2] + vector[2]
    return ([x,y,z],color[0],color[1],color[2])


def findPlane(components) :
    #ADD POINTS TO COMPONENTS
    origin = ([0,0,0],255,255,255)
    components = 5 * np.array(components)
    pt1 = addVector(origin,components[0])
    pt2 = addVector(origin,components[1])
    # pt3 = addVector(addVector(origin,components[0]),components[1])
    # return np.array(
    #     [origin,pt1,pt2,pt3], \
    #     dtype =[('vertex_indices', 'i4', (3,)), \
    #     ('red', 'u1'), ('green', 'u1'), \
    #     ('blue', 'u1')]
    #     )
    return np.array(
    [origin,pt1,pt2], \
    dtype =[('vertex_indices', 'i4', (3,)), \
    ('red', 'u1'), ('green', 'u1'), \
    ('blue', 'u1')]
    )


    # .astype([('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])



if __name__ == "__main__":
    plydata = PlyData.read('DESK_OUTPUT/reconstruction_sequential/robust.ply')
    # print("vertex",plydata["face"])
    a = plydata.elements[0].data 
    components = automaticPlaneFinding(a)   
    plane = findPlane(components)
    print("WTF",a)
    # face = plane.astype([('vertex_indices', 'i4', (3,)), \
    #     ('red', 'u1'), ('green', 'u1'), \
    #                         ('blue', 'u1')])

    el = PlyElement.describe(plane, 'face')
    print(el.data,type(el.data),el)
    PlyData([plydata.elements[0],el]).write('some_binary.ply')


# plydata.close()