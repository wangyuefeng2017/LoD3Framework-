# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:09:26 2020

@author: wpc
"""

import readgml1
import initpara_batch3_stage as initpara_batch3
import creat1
from lxml import etree
import cv2
# import rect_im
import joblib
import os
import amendwindow22_stage_PATCH as optimal

DIRECTORY=r'C:\Users\wpc\Desktop\lod223\lod3'


ns_citygml = "http://www.opengis.net/citygml/2.0"

ns_gml = "http://www.opengis.net/gml"
ns_bldg = "http://www.opengis.net/citygml/building/2.0"
ns_tran = "http://www.opengis.net/citygml/transportation/2.0"
ns_veg = "http://www.opengis.net/citygml/vegetation/2.0"
ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
ns_xAL = "urn:oasis:names:tc:ciq:xsdschema:xAL:2.0"
ns_xlink = "http://www.w3.org/1999/xlink"
ns_dem = "http://www.opengis.net/citygml/relief/2.0"
ns_fme = "http://www.safe.com/xml/xmltables"

ns_app = "http://www.opengis.net/citygml/appearance/2.0"
ns_gen = "http://www.opengis.net/citygml/generics/2.0"

nsmap = {
    None: ns_citygml,
    'gml': ns_gml,
    'bldg': ns_bldg,
    'tran': ns_tran,
    'veg': ns_veg,
    'xsi': ns_xsi,
    'xAL': ns_xAL,
    'xlink': ns_xlink,
    'dem': ns_dem,
    'fme': ns_fme,
    'app' : ns_app,
    'gen' : ns_gen
}

def buildinginformationforgml(gmlfile):
    CITYGML = etree.parse(gmlfile)
    root = CITYGML.getroot()
    cityObjects = []
    buildings = []

    
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
        buildingclasses.append(readgml1.Building(b, id))
    
    count1 = 0
    count2 = 0
    for bu in buildingclasses:
        imageinf=bu.imageinformation
        wallinf=bu.posinformation
        # print ('Iterating the ',ct,'-th building')
        # ct+=1
        for k in  wallinf.keys():
            # LinearRing=wallinf[k]['pos'].text
            # listPoints=readgml1.transpos(LinearRing)
            count1 += 1
            if k in imageinf.keys():# and float(wallinf[k]['area'].text) > 10:
                count2+=1
    print ('cccccc',count1,  count2)
    print ("\tI have read all buildings, now I will search for the images corresponding to facade...")
    
    return buildingclasses,buildings

def Obtainrectimageforgml(gmlfile,savepath,savepath2):
    CITYGML = etree.parse(gmlfile)
    root = CITYGML.getroot()
    cityObjects = []
    buildings = []

    
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
        buildingclasses.append(readgml1.Building(b, id))
    
    print ("\tI have read all buildings, now I will search for the images corresponding to facade...")

    ct=0

    rel={}
    #-- Iterate all buildings
    for bu in buildingclasses:
        imageinf=bu.imageinformation
        wallinf=bu.posinformation
        print ('Iterating the ',ct,'-th building')
        ct+=1
        for k in  wallinf.keys():
            LinearRing=wallinf[k]['pos'].text
            listPoints=readgml1.transpos(LinearRing)
            
            if k in imageinf.keys():# and float(wallinf[k]['area'].text) > 10:
                imname='../'+imageinf[k]['imgname'].text
                # print (imname)
                imgop=cv2.imread(imname)
                s=imgop.shape
                imring=imageinf[k]['coord'].text
                coordlist=readgml1.transcoord(imring,s)
                imagedraw=readgml1.clipimage(imgop,coordlist)
                warped,scale = readgml1.rect_im.four_point_transform(imgop, listPoints, coordlist)
                # print (warped.shape)

                if len(warped) > 0:
                    imnamesave = k+imname.split('/')[-1]

                    cv2.imwrite(savepath+imnamesave,warped)
                    cv2.imwrite(savepath2+imnamesave,imagedraw)
                    
                    rel[k] = {'img':imnamesave,'scale':scale}
        
    joblib.dump((rel),'img_model.pkl')

    print ("All done.")
    return buildingclasses,buildings,rel

def obtainparaset(path_read,path_save,rel):

    # path_read = r'result3'
    # path_save = r'save_para'
    ct=0
    # rel = joblib.load('img_model.pkl')
    for value1 in rel.values():
        img = value1['img']
        imgname = img.split('/')[-1].split('.')[0]
        para_path = path_read + '/' + imgname + '.pkl'
        image_path = path_read + '/' + imgname+'_splash.png'
        scale = value1['scale']

        para_set_create = initpara_batch3.get_para(image_path, para_path, scale)

        joblib.dump(para_set_create, path_save+'/'+imgname+'.pkl')

        
        print ('processing the ',ct,'-th image')
        ct+=1


def writelod3(suffix,para_path,FULLPATH,buildingclasses,buildings,rel_model):
    CityGMLs = {}
    
    # suffix = 'LOD3.3'
    CityGMLs[suffix] = creat1.createCityGML(suffix)
    

    for i in range(len(buildingclasses)):
        bldinf = buildingclasses[i]
        id = buildings[i].attrib['{%s}id' %ns_gml]
        print (id)
        # bldinf = Building(b, id)
        creat1.CityGMLbuildingLOD3Semantics(buildings[i], CityGMLs[suffix], bldinf, rel_model, para_path, id)#, attributes)
        print ('creating the ',i,'-th building')    
    
    creat1.storeCityGML(CityGMLs,suffix)


#%% ------------------First step: 3.2.1 Obtaining LoD2 Shell Model and Facade Texture Image---------------
if __name__ == '__main__':
    gmlfile = "../export.gml"
    savepath = 'appearence_rect2/'
    savepath2 = 'appearence_draw2/' 
    
    if os.path.exists(savepath) is not True:
        os.mkdir(savepath)
    
    if os.path.exists(savepath2) is not True:
        os.mkdir(savepath2)
    
    buildingclasses,buildings,rel = Obtainrectimageforgml(gmlfile,savepath,savepath2)
    
#%% --------Second step: 3.2.2 Reconstruction of Semantic Facade Models--

# The step of Instance Segmentation is excuted by Mask RCNN, after that, bbox.png and .pkl of facade image can be output.
# --waiting for the result of Mask-RCNN----

# if wanna adjust lod, just run this term(buildinginformationforgml)
    buildingclasses,buildings = buildinginformationforgml(gmlfile)
    rel = joblib.load('img_model.pkl')
    segment_read = r'ces2'
    para_save = r'save_para'
    if os.path.exists(para_save) is not True:
        os.mkdir(para_save)

# The step of Regularization can be excuted by following code.

    obtainparaset(segment_read,para_save,rel)

    para_adjust = r'adjust_para2'
    if os.path.exists(para_adjust) is not True:
        os.mkdir(para_adjust)

# The step of Optimization can be excuted by following code.
      
    optimal.optimalwindows(segment_read,savepath,rel,para_adjust)

#%% --------Third step:3.2.3. Exporting: LoD3 Model Generation-------------
    suffix = 'LOD3.5'
    writelod3(suffix,para_adjust,gmlfile,buildingclasses,buildings,rel)
    
    # para_adjust = r'adjust_para2'
    # writelod3(suffix,bldinf,relfile, para_adjust)
    
    
    
    
    
    
    
    
    
    
    
    





