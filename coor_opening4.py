# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:01:49 2020

@author: wpc
"""

import math
import numpy as np
import matplotlib.pyplot as plt 
import copy
# import os
import joblib
# import skimage
from PIL import Image


def window_rect_complex(H,para_label): # label--> 'window','balcony','door'......
    p_component=[]
    num_floor=0
    h_sp=H
    
    eachfloor=para_label['eachfloor']
    for i in range(len(eachfloor)):
        num_floor+=eachfloor[str(i)]['floor']['sf']
        # print (num_floor)
        for nf in range(1,eachfloor[str(i)]['floor']['sf']+1):
            h_sp-=eachfloor[str(i)]['floor']['hf']
            h_gap=h_sp+eachfloor[str(i)]['h_gap']
            # print ('nf',nf,h_sp)
            w_sp=eachfloor[str(i)]['w_bound']
            nv=-1
            nm=0
            num_eachwindow=eachfloor[str(i)]['num_eachwindow']
            edge=eachfloor[str(i)]['edge']
            nwindow=eachfloor[str(i)]['nwindow']
            if eachfloor[str(i)]['num_vertices']>1:
                for n in range(nwindow):
                    if nm<n<num_eachwindow[str(nv+1)]['sw']+nm:
                        w_sp+=num_eachwindow[str(nv+1)]['ww']+edge['d_intra'][str(2*(nv+1))]
                    elif n==num_eachwindow[str(nv+1)]['sw']+nm:
                        nm+=num_eachwindow[str(nv+1)]['sw']
                        # print ('aaaaaaaaa',nm,edge['d_inter'],num_eachwindow)
                        w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                        nv+=1
                    p_component.append(((w_sp,h_gap),num_eachwindow[str(nv+1)]['ww'],num_eachwindow[str(nv+1)]['hw']))
                    
            else:
                for n in range(nwindow):
                    w_sp=eachfloor[str(i)]['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
                    p_component.append(((w_sp,h_gap),num_eachwindow['0']['ww'],num_eachwindow['0']['hw'])) 
                    
    return p_component

def transfd(dshape,z=0.5):
    verts=[]
    for window in dshape:
        # vert=[]
        pos,w,h=window
        x,y=pos
        vert=[(x,z,y),(x+w,z,y),(x+w,z,y+h),(x,z,y+h),(x,z,y)] #front
        vert2=[(x,z,y),(x+w,z,y),(x+w,0,y),(x,0,y),(x,z,y)] #bottom
        vert3=[(x,z,y),(x,0,y),(x,0,y+h),(x,z,y+h),(x,z,y)] #left
        vert4=[(x+w,z,y),(x+w,0,y),(x+w,0,y+h),(x+w,z,y+h),(x+w,z,y)] #right
        vert5=[(x,0,y),(x+w,0,y),(x+w,0,y+h),(x,0,y+h),(x,0,y)] # behind
        vert6= [(x,z,y+h),(x+w,z,y+h),(x+w,0,y+h),(x,0,y+h),(x,z,y+h)] # top
        # if lab=='facade':
        #     verts.append([vert2,vert3,vert4,vert5,vert6])
        # elif lab=='window' or lab=='balcony':
        verts.append([vert,vert2,vert3,vert4,vert5,vert6])
        # elif lab=='corrid':
        #     verts.append([vert,vert2,vert3,vert4,vert6])
    
    return verts

# def get_vertices(p_component):
#     for c in p_component:
        # verts=transfd(p_component,0.5,'window')

def rotator(vertex, sine, cos, tan, origin_of_rotation):
    "Rotate the vertex around the origin by an angle (2D). Cos and sin are already precomputed to make the calculations more efficient due to many repetitions."
    vertex = [float(vertex[0]), float(vertex[1]), float(vertex[2])]
    rotated = [None, None, vertex[2]+origin_of_rotation[2]]
    # rotated[0] = ((vertex[0]) * cos - (vertex[1]) * sine) + origin_of_rotation[0]
    # rotated[1] = ((vertex[0]) * sine + (vertex[1]) * cos) + origin_of_rotation[1]
    
    l = np.sqrt((vertex[0])**2+(vertex[1])**2)
    xm = l/(np.sqrt(1+tan**2))*(tan/(abs(tan)+0.001))
    rotated[0] = xm + origin_of_rotation[0]
    rotated[1] = xm*tan + origin_of_rotation[1]
    
    return rotated



def get_rotation(modelpts, points_to_rotate):
    newmp=[]
    zs=[]
    for x,y,z in modelpts:
        # newmp.append((x+y,z))
        newmp.append((np.sqrt(x*x+y*y),z))
        zs.append(z)
    newmp=np.array(newmp)
    
    s = newmp.sum(axis = 1)
    
    # diff = np.diff(newmp, axis = 1)

    x1,y1,z1 = modelpts[np.argmin(s)]
    x2,y2,z2 = modelpts[np.argmax(s)]
    z1 = min(zs)
    angle_of_rotation = math.atan((y2-y1)/((x2-x1)+0.001))
    
    radian_rotation = math.radians(angle_of_rotation)
    sine_rotation = math.sin(radian_rotation)
    cosine_rotation = math.cos(radian_rotation)
    tan_rotation = (y2-y1)/((x2-x1)+0.001)
    
    origin_coords = (x1,y1,z1)
    # print (origin_coords)
    new_w = []
    for point_to_rotate in points_to_rotate:
        new_f = []
        for v in point_to_rotate:
            new_v = []
            for p in v :
            # print (v)
                rotated_point = rotator(p, sine_rotation, cosine_rotation, tan_rotation, origin_coords)
                new_v.append(rotated_point)
            new_f.append(new_v)
        new_w.append(new_f)
    return new_w

def transpos(LinearRing):
    listPoints=[]
    lr=LinearRing.split()
    assert(len(lr) % 3 == 0)
    for i in range(0, len(lr), 3):
        listPoints.append((float(lr[i]), float(lr[i+1]), float(lr[i+2])))
    return listPoints

def get_coord(para, modelpts):
    
    W=para['W']
    H=para['H']
    para_set=para['para_set']
    modelinf={}
    # modelinf['W']=W
    # modelinf['H']=H
    for label in list(para_set.keys()):
        p_component=window_rect_complex(H,para_set[label])
        
        p_verts=transfd(p_component)
        if len(p_verts) > 0:
            rotated_p_component = get_rotation(modelpts, p_verts)
            modelinf[label]=rotated_p_component

        else:
            modelinf[label]= []
        
    return modelinf

def addfig(ax,c,a,verts):
    for vert in verts:
        # poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['r']), alpha=1)
        poly = mpl3.art3d.Poly3DCollection(vert[0:4],facecolors=c, alpha=a)
        ax.add_collection3d(poly)
        
        
if __name__ == '__main__':
    para_path = r'save_para'
    imgname = 'tex_2536667'
    para = joblib.load(para_path+'/'+imgname+'.'+'pkl')
    LinearRing = '2.5496955937E7 6672468.666 16.46 2.5496956309E7 6672460.868 16.46 2.5496956349E7 6672460.027 16.46 2.5496956355E7 6672459.907 16.46 2.5496956355E7 6672459.907 34.044 2.5496956349E7 6672460.027 33.98 2.5496956349E7 6672460.027 38.28 2.5496956309E7 6672460.868 38.719 2.5496955937E7 6672468.666 38.724 2.549695591E7 6672469.231 38.725 2.549695591E7 6672469.231 16.46 2.5496955937E7 6672468.666 16.46'
    modelpts = transpos(LinearRing)
    modelinf = get_coord(para, modelpts)
    
    import mpl_toolkits.mplot3d as mpl3
    fig = plt.figure()
    ax = mpl3.Axes3D(fig)
    
    addfig(ax,['b'],1,modelinf['window'])
    # addfig(ax,['purple'],1,vertsbal)
    # addfig(ax,['w'],0.6,vertsf1)
    # addfig(ax,['w'],0.6,vertsf2)
    # addfig(ax,['w'],0.6,vertsf3)
     
    # for vert in verts:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['r']), alpha=1)
    #     ax.add_collection3d(poly)
        
    # for vert in vertsbal:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['g']), alpha=0.6)
    #     ax.add_collection3d(poly)
        
    # for vert in vertsf:
    #     poly = mpl3.art3d.Poly3DCollection(vert,facecolors=np.random.choice(['y']), alpha=0.6)
    #     ax.add_collection3d(poly)
    x1,y1,z1 =(25496956.355, 6672459.907, 16.46)
    ax.set_xlim3d(left=x1-5,right=x1+20)
    ax.set_ylim3d(bottom=y1-5, top=y1+20)
    ax.set_zlim3d(bottom=z1-2,top=z1+12)
    # ax.set_aspect(1)
    plt.show()
    plt.close()




















