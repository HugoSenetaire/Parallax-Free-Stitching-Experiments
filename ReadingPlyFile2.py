import pymesh
# import pymesh.load_mesh
import sklearn
import numpy as np
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

# import tkinter as tkinter
# import tkinter as tk

from Tkinter import *
import Tkinter as tk
from tkFileDialog import askopenfilename
from sys import platform as sys_pf
from PIL import ImageTk
from PIL import Image

# Problem with MACOSX Use TKAGG as backend
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.tkagg as tkagg
import matplotlib.pyplot as plt

## PREPROCESSING 1 : FIND PROJECTION PLANE 

def findPhotoPosition(data,colorPhoto = [0,255,0]):
    photoPos = []
    for point in data : 
        if np.all([point[3],point[4],point[5]]==colorPhoto):
            photoPos.append([point[0],point[1],point[2]])
    return photoPos



def automaticPlaneFinding(data):
    
    pca = PCA(n_components = 2)
    # print(photoPos)
    pca.fit(photoPos)

    return pca.components_

def addVector(point,vector):
    x = point[0] + vector[0]
    y = point[1] + vector[1]
    z = point[2] + vector[2]
    return ([x,y,z])


def drawPlane(components, origin = [0,0,0]) :
    #ADD POINTS TO COMPONENTS

    components = 5 * np.array(components)
    pt1 = addVector(origin,components[0])
    pt2 = addVector(origin,components[1])
    pt3 = addVector(addVector(origin,components[0]),components[1])
    # return np.array(
    #     [origin,pt1,pt2,pt3], \
    #     dtype =[('vertex_indices', 'i4', (3,)), \
    #     ('red', 'u1'), ('green', 'u1'), \
    #     ('blue', 'u1')]
    #     )
    return np.array([origin,pt1,pt2])


    # .astype([('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

def orthogonalProduct(u,v):
    newVector = [
        u[1]*v[2]- u[2]*v[1],
        u[2]*v[0]- u[0]*v[2],
        u[0]*v[1]- u[1]*v[0],
        ]
    return newVector


def scalarProduct(u,v):
    return u[1]*v[1]+u[2]*v[2]+u[0]*v[0]


def projection(x,compo1,compo2):
    return [scalarProduct(x,compo1),scalarProduct(x,compo2)]


def normalize(x):
    return np.array(x) / np.linalg.norm(x)

## PREPROCESSING 2 :  PICTURE SURFACE SELECTION


def drawAll(filename):
    """ Draw the different lines that will define the surface """
    



    root = Tk()
    global points
    global realPoints
    points = []
    realPoints = []
    valueShift = []


    # def draw_figure(canvas, figure, loc=(0, 0)):
    #     """ Draw a matplotlib figure onto a Tk canvas

    #     loc: location of top-left corner of figure on canvas in pixels.
    #     Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    #     """
        
    #     figure_canvas_agg = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
    #     figure_canvas_agg.draw()
    #     figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    #     figure_w, figure_h = int(figure_w), int(figure_h)
    #     photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    #     # Position: convert from top-left anchor to center anchor
    #     # canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)
    #     canvas.create_image(0,0,image=photo)

    #     # Unfortunately, there's no accessor for the pointer to the native renderer
    #     tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    #     # Return a handle which contains a reference to the photo object
    #     # which must be kept live or else the picture disappears
    #     return photo

    

    # setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)


    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set,
                    yscrollcommand=yscroll.set,
                    width=1200, height=750)
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)
    

    # A modifier pour tracer directement en matplot lib
    img = ImageTk.PhotoImage(Image.open(filename).convert('RGB'))
    global shape
    shape = np.shape(Image.open(filename))
    print(shape)
    canvas.create_image(0, 0, image=img, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    
    def updateWindow():
        # draw_figure(canvas,fig)
        canvas.create_image(0, 0, image=img, anchor="nw")
        canvas.config(scrollregion=canvas.bbox(ALL))
        for coords_idx in range(len(points)):
            coords = points[coords_idx]
            canvas.create_rectangle(coords[0] - 5, coords[1] - 5,
                                    coords[0] + 5, coords[1] + 5, fill="yellow")
  
            if coords_idx < len(points) - 1:
                canvas.create_line(points[coords_idx][0], points[coords_idx][1], 
                    points[(coords_idx+1)][0], points[(coords_idx+1)][1])

      
    # Functions called on different events
    def printcoords(event):
        """ Function called when left click on image """
        print("TEST")
        global points
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
    
        points.append(np.array([x, y]))
        realPoints.append(np.array([x/shape[0],y/shape[1]]))
        updateWindow()

    def _on_mousewheel_y(event):
        canvas.yview_scroll(-1*event.delta, "units")

    def _on_mousewheel_x(event):
        canvas.xview_scroll(-1*event.delta, "units")

    def end_and_save():

        root.destroy()


    def delete_last_point():
        if len(points) > 0:
            del points[-1]
            updateWindow()

    def delete_all_points():
        del points[:]
        updateWindow()





    # Events and layouts
    bottom = Frame(root)
    bottom.pack(side=BOTTOM, fill=BOTH, expand=True)
    canvas.bind("<Button 1>", printcoords)
    canvas.bind("<MouseWheel>", _on_mousewheel_y)
    canvas.bind("<Shift-MouseWheel>", _on_mousewheel_x)
    keep_button = Button(root, text="Save current line", command=end_and_save)
    delete_button = Button(root, text="Delete last point", command=delete_last_point)
    delete_all_button = Button(root, text="Delete all points", command=delete_all_points)
    keep_button.pack(in_=bottom, side=LEFT)
    delete_button.pack(in_=bottom, side=LEFT)
    delete_all_button.pack(in_=bottom, side=LEFT)

    root.mainloop()
    
    return realPoints


# A faire directement en matplotlib

# def drawAllPlt(projectedVertices):
#     fig = plt.figure()
#     ax = fig.subplot    






## PIPELINE

if __name__ == "__main__":
    mesh = pymesh.load_mesh('DESK_OUTPUT/reconstruction_sequential/robust.ply')
    # mesh, info = pymesh.remove_isolated_vertices(mesh)
    vertices = mesh.vertices
    plydata = PlyData.read('DESK_OUTPUT/reconstruction_sequential/robust.ply')
    a = plydata.elements[0].data 


    photoPos = findPhotoPosition(a)


    components = automaticPlaneFinding(a)   
    orthogonal_vector = orthogonalProduct(components[0],components[1])
    
    compoPhotos = automaticPlaneFinding(photoPos)

    # new_components = [orthogonal_vector,compoPhotos[0]]

    vector_aux = orthogonalProduct(orthogonal_vector,compoPhotos[0])
    
    new_components = [normalize(compoPhotos[0]),normalize(orthogonalProduct(compoPhotos[0],vector_aux))]



    projectedVertices = []
    for vertex in vertices :
        projectedVertices.append(projection(vertex,new_components[0],new_components[1]))


    filename = "tempFile.jpg"
    fig = plt.figure(1)
    ax =  fig.add_subplot(1,1,1)
    ax.scatter(np.transpose(projectedVertices)[0],np.transpose(projectedVertices)[1])
    ax.axis('off')
    fig.savefig(filename)

    plt.show()

    points = drawAll(filename)
    plt.show()
    # facePoints = drawPlane(new_components,origin = vertices[-1])
    
    print(points)

    # face = drawPlane(new_components,origin = vertices[-1])
    # vertices = np.concatenate((vertices,face))
    # aux = len(vertices)
    # faces = np.array([[aux-1,aux-2,aux-3]])
    # mesh2 = pymesh.form_mesh(vertices, faces)
    # pymesh.save_mesh("test.obj", mesh2);

# plydata.close()