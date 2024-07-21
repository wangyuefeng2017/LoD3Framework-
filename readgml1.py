# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:02:29 2020

@author: wpc
"""
from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mpl3
import cv2
# from PIL import Image
from skimage import draw,data
import random
import rect_im
import joblib

ns_citygml = "http://www.opengis.net/citygml/2.0"

ns_gml = "http://www.opengis.net/gml"
ns_bldg = "http://www.opengis.net/citygml/building/2.0"
ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
ns_xAL = "urn:oasis:names:tc:ciq:xsdschema:xAL:2.0"
ns_xlink = "http://www.w3.org/1999/xlink"
ns_dem = "http://www.opengis.net/citygml/relief/2.0"
ns_app = "http://www.opengis.net/citygml/appearance/2.0"
ns_gen = "http://www.opengis.net/citygml/generics/2.0"

nsmap = {
    None : ns_citygml,
    'gml': ns_gml,
    'bldg': ns_bldg,
    'xsi' : ns_xsi,
    'xAL' : ns_xAL,
    'xlink' : ns_xlink,
    'dem' : ns_dem,
    'app' : ns_app,
    'gen' : ns_gen
}


class Building(object):
    def __init__(self, xml, id):
        #-- ID of the building
        self.id = id
        #-- XML tree of the building
        self.xml = xml
        #-- Data for each roof surface required for the computation of the solar stuff
        self.roofdata = {}
        #-- List of IDs of openings, not to mess with usable roof surfaces
        self.listOfOpenings = []

        self.posinformation=self.posext()
        self.imageinformation=self.imageext()

    
    def posext(self):

        self.walls = []
        self.wallsurfaces = []
        # wallarea = 0.0
        # openings = 0.0
        self.poslistset={}
        self.area=[]
        self.posid=[]
        self.pospos=[]
        #-- Account for openings
        for child in self.xml.getiterator():
            if child.tag == '{%s}WallSurface' %ns_bldg:
                self.walls.append(child)
                # openings += oparea(child)
        for surface in self.walls:
            for w in surface.findall('.//{%s}Polygon' %ns_gml):
                self.wallsurfaces.append(w)
            
        # for wallsurface in self.wallsurfaces:
                for child in w.getiterator():
                    if child.tag == '{%s}LinearRing' %ns_gml:
                        # print (child.attrib['{%s}id' %ns_gml])
                        # self.pospos.append(child)
                        # self.posid.append(child.attrib['{%s}id' %ns_gml])
                        self.poslistset[child.attrib['{%s}id' %ns_gml]]={'pos': child.find('.//{%s}posList' %ns_gml)}

        
        return self.poslistset
    
    def imageext(self):
        """"extracting image information, which contains id, coordinates, name"""
        self.imageinfset={}
        self.app=[]
        self.appsurf=[]
        # self.imageid=[]
        self.imagename=[]
        self.imagecoordinates=[]
        for child in self.xml.getiterator():
            if child.tag == '{%s}surfaceDataMember' %ns_app:
                self.app.append(child)
                self.imagename.append(child.find('.//{%s}imageURI' %ns_app))
                
        # for surface in self.app:
                for w in child.findall('.//{%s}target' %ns_app):
                    self.appsurf.append(w)
                
                    # for imsurf in self.appsurf:
                    for child2 in w.getiterator():
                        if child2.tag == '{%s}TexCoordList' %ns_app:
                            # print (child.attrib['{%s}id' %ns_gml])
                            self.imageinfset[child2.find('.//{%s}textureCoordinates' %ns_app).attrib['ring'][1:]]={'coord':child2.find('.//{%s}textureCoordinates' %ns_app),'imgname':child.find('.//{%s}imageURI' %ns_app)}

        
        return self.imageinfset

def clipimage(image,vert):
    roi_vert = np.asarray(vert)
    roi_vert = np.expand_dims(roi_vert, axis=0)

    xset=[]
    yset=[]
    for x,y in vert:
        xset.append(x)
        yset.append(y)
    cc,rr=draw.polygon_perimeter(yset,xset)
    draw.set_color(image,[cc,rr],[255,0,0])

    return image

def transpos(LinearRing):
    listPoints=[]
    lr=LinearRing.split()
    assert(len(lr) % 3 == 0)
    for i in range(0, len(lr), 3):
        listPoints.append((float(lr[i]), float(lr[i+1]), float(lr[i+2])))
    return listPoints

def transcoord(ring,s):
    coordlist=[]
    cl=ring.split()
    sc,sr=s[0:2]
    assert(len(cl) % 2 == 0)
    for i in range(0, len(cl), 2):
        coordlist.append(((float(cl[i]))*sr, (1-float(cl[i+1]))*sc))
    return coordlist

def addfig(c,a,verts):
    fig = plt.figure()
    ax = mpl3.Axes3D(fig)
    # for vert in verts:
    poly = mpl3.art3d.Poly3DCollection(verts,facecolors=c, alpha=a)
    ax.add_collection3d(poly)
    x,y,z=verts[0][0]     
    ax.set_xlim3d(left=x-20,right=x+20)
    ax.set_ylim3d(bottom=y-20, top=y+20)
    ax.set_zlim3d(bottom=z-1,top=z+20)
            
    plt.show()
    plt.close()

# def rectimg(image,listPoints,coordlist):
    
    
if  __name__ == "__main__":
# DIRECTORY="G:\FME_36365D5B_1606484671488_8764\FILECOPY_1\TEMP\2549_6672"
# for f in glob.glob("*.gml"):
#     FILENAME = f[:f.rfind('.')]
# while True:
    FULLPATH = "../3910_5821.gml"#DIRECTORY + f
    
    CITYGML = etree.parse(FULLPATH)
    root = CITYGML.getroot()
    cityObjects = []
    buildings = []
    
    listofxmlroofsurfaces = []
    roofsurfacedata = {}
    
    #-- Find all instances of cityObjectMember and put them in a list
    for obj in root.getiterator('{%s}cityObjectMember'% ns_citygml):
        cityObjects.append(obj)
    
    # print (FILENAME)
    print ("\tThere are", len(cityObjects), "cityObject(s) in this CityGML file")
    
    for cityObject in cityObjects:
        for child in cityObject.getchildren():
            if child.tag == '{%s}Building' %ns_bldg:
                buildings.append(child)
    
    #-- Store the buildings as classes
    buildingclasses = []
    for b in buildings:
        id = b.attrib['{%s}id' %ns_gml]
        # print (id)
        buildingclasses.append(Building(b, id))
    
    print ("\tI have read all buildings, now I will search for roofs and estimate their solar irradiation...")
    
    #-- Store the obtained data in a dictionary
    
    
    #-- Check if there are roof surfaces in the file
    #%%
    rsc = 0
    ct=0
    savepath='appearence_rect2/'
    savepath2='appearence_draw/'
    # fig = plt.figure()
    # ax = mpl3.Axes3D(fig)
    color=['r','g','b','c','y']
    imname2='im'
    mulit=0
    rel={}
    #-- Iterate all buildings
    for bu in buildingclasses:
        imageinf=bu.imageinformation
        wallinf=bu.posinformation
        print (ct)
        ct+=1
        for k in  wallinf.keys():
            LinearRing=wallinf[k]['pos'].text
            listPoints=transpos(LinearRing)
            
            addfig([random.choice(color)],1,[listPoints])

    
    print ("All done.")