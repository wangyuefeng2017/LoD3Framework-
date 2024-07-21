# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 21:44:57 2022

@author: wyf
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 00:38:16 2022

@author: wyf
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:52:10 2022

@author: wyf
"""

import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt 
import inverse_facade_stage as inverse_facade
import joblib
# import skimage
from PIL import Image
import initpara_batch3_stage as initpara_batch3
import cv2
import os
import copy
# def fill_hole(num_window):

    
# def window_medium(p_window):
range_para={'h_floor':(2.0,3.5),
            'h_first':(3.0,5.5),
            'h_roof':(0.1,1),
            'h_gap':(0.8,2.5),
            'h_fgap':(0.0,2.5),
            'h_window':(0.82,2.8),
            'w_window':(0.5,2.5),
            'w_sp':(0.5,2.0),
            'w_bound':(0.5,2.5)}

def cal_iou(rectangles_from_segmentation, rectangles_after_regularization,s):
    def calculate_iou(rect1, rect2):
        # rect: ((x, y), w, h)
        (x1, y1), w1, h1 = rect1
        (x2, y2), w2, h2 = rect2
    
        # 计算交集区域
        x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection_area = x_intersection * y_intersection
    
        # 计算并集区域
        union_area = w1 * h1 + w2 * h2 - intersection_area
    
        # 计算IOU
        iou = intersection_area / union_area if union_area > 0 else 0
    
        return iou
    
    def calculate_average_iou(rectangles1, rectangles2):
        total_iou = 0
        total_pairs = 0
        for rect2set in rectangles2.values():
            
            for rect1 in rectangles1:
                for rect2 in rect2set:
                    iou = calculate_iou(rect1, rect2)
                    if iou>0:
                        total_iou += iou
                        total_pairs += 1
    
        average_iou = total_iou / total_pairs if total_pairs > 0 else 0
        return average_iou
    
    # # 示例用法
    # rectangles_from_segmentation = [((x1, y1), w1, h1), ((x2, y2), w2, h2), ...]  # 实例分割结果
    # rectangles_after_regularization = [((x3, y3), w3, h3), ((x4, y4), w4, h4), ...]  # 规则化后的结果
    
        # 计算平均IoU
    average_iou = calculate_average_iou(rectangles_from_segmentation, rectangles_after_regularization)
    
    print("Average IoU:", average_iou)

def stat(p_window2):
    wset = []
    hset = []
    for (x,y),w,h in p_window2:
        wset.append(np.around(w,2))
        hset.append(np.around(h,2))
    
    hwed = np.median(hset)
    wmed = np.median(wset)
    wavg = np.average(wset)
    wcount = stats.mode(wset)[0][0]
    return [hwed, wmed, wavg, wcount]

# if abs(w)



def getvalue(wh,k):
    
    def getinit(wh,r):
        outputarray = []
        
        rmax = np.around(wh/r[0])     
        rmin = np.around(wh/r[1])
        # print (wh, r, rmax, rmin)
        if rmax > rmin:
            numk = np.random.randint(rmin,rmax)
            initvalue = round(wh/numk, 2)
            for i in range(0,numk-1):
                outputarray.append(initvalue)
            lastvalue = round(wh-(numk-1)*initvalue,2)
            outputarray.append(lastvalue)
        else:
            numk = rmax
            outputarray.append(wh)

        return outputarray

    if k == 'h_first' and wh > 10:
        hfirst = round(np.random.uniform(range_para['h_first'][0],range_para['h_first'][1]),2)
        hleft = wh - hfirst
        r = (range_para['h_floor'][0],hfirst)
        outputarray = getinit(hleft,r)
        outputarray.append(hfirst)
        return outputarray
    
    else:
        r = range_para[k]
        return getinit(wh,r)

def window_rect_complex(W,H,para_set,label,s=1): # label--> 'window','balcony','door'......
    p_component=[]
    num_floor=0
    h_sp=H
    
    eachfloor=para_set[label]['eachfloor']
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
                        try:
                            w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                        except:
                            # print (eachfloor[str(i)])
                            sys.exit()
                        nv+=1
                    p_component.append(((w_sp/s,h_gap/s),num_eachwindow[str(nv+1)]['ww']/s,num_eachwindow[str(nv+1)]['hw']/s))
                    
            else:
                for n in range(nwindow):
                    w_sp=eachfloor[str(i)]['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
                    p_component.append(((w_sp/s,h_gap/s),num_eachwindow['0']['ww']/s,num_eachwindow['0']['hw']/s)) 
    
    return p_component


def plot(title,W,H,para_set_create,p_window): 
    
    def plot_all(ax,p_component,W,H,c):
    
        for (x,y),w,h in p_component:
            rect1=plt.Rectangle((x,y),w,h,color=c)
            plt.gca().add_patch(rect1)
        
    p_window2=window_rect_complex(W,H,para_set_create,'window')  
    p_Stairwell_window=window_rect_complex(W,H,para_set_create,'Stairwell_window')  
    p_balcony3=window_rect_complex(W,H,para_set_create,'balcony')
    p_door4=window_rect_complex(W,H,para_set_create,'door')
    
    fig=plt.figure(dpi=120)
    ax = fig.add_subplot(111)
    
    # plt.title(title+str(len(p_window)));
    plt.title("opt d% afloor %s" % (title[0],title[1]))#+str(len(p_window)));
    # print (title+' '+str(len(p_window2)))
    
    plot_all(ax,p_window2,W,H,'blue')
    plot_all(ax,p_window,W,H,'red')
    # plot_all(ax,p_balcony3,W,H,'red')
    # plot_all(ax,p_door4,W,H,'green')
    # plot_all(ax,p_window2,W,H,0,0,'blue')
    plot_all(ax,p_balcony3,W,H,'red')
    plot_all(ax,p_door4,W,H,'green')
    plot_all(ax,p_Stairwell_window,W,H,'cyan')
    
    ax.set_xlim(0-1,0+W+2)
    ax.set_ylim(0-1,0+H+2)
    
    ax.set_aspect(1)
    
    plt.show()
    # return p_window2


def adj_floor(H, hfset, hspset, label):
    
    # hfnew = []
    # hmed = np.median(hfset)
    labelnew = copy.deepcopy(label)
    hspm = []
    # label['eachfloor'][str(len(hspset)-1)] = label['eachfloor'][str(len(hspset)-2)] 
    # hfsetnew = []
    j=0
    
    for hsp, h in hspset:
        if hsp>range_para['h_gap'][0]:
            hspm.append(hsp)
    # hspmin = np.max([np.min(hspm), np.min(hfset)])
    hspmin = np.min(hspm)
    # if hspmin<range_para['h_gap'][0]:
    #     hspmin = range_para['h_gap'][0]
    # print ('hfset', hfset, hspset, hspmin)
    for i in range(len(hspset)-1):
        if (hspset[i][0]-hspset[i][1])/2 > range_para['h_gap'][0]:
            hspmin = (hspset[i][0]-hspset[i][1])/2
        if 2*hspset[i][1]+hspmin < hfset[i]-range_para['h_gap'][0]:# and (hspset[i][0]-hspset[i][1])/2 >= range_para['h_gap'][0]:
            # print ('debug', hspmin, (hspset[i][0]-hspset[i][1])/2)
            # hfsetnew[j] = hfset[i]-hspset[i][1]-hspmin
            # if (hspset[i][0]-hspset[i][1])/2 >= range_para['h_gap'][0]:
            # label['eachfloor'][str(j)]['floor']['hf'] = hfset[i]-hspset[i][1]-hspmin
            labelnew['eachfloor'][str(j)] = copy.deepcopy(label['eachfloor'][str(i)])
            labelnew['eachfloor'][str(j)]['floor']['hf'] = hfset[i]-hspset[i][1]-hspmin
            # print ('a1',i, j, labelnew['eachfloor'][str(j)]['floor']['hf'], label['eachfloor'][str(i)]['floor']['hf'])
            # hfsetnew[j+1] = hfset[j]-hfsetnew[j]
            labelnew['eachfloor'][str(j+1)] = copy.deepcopy(label['eachfloor'][str(i)])
            labelnew['eachfloor'][str(j+1)]['floor']['hf'] = hspset[i][1]+hspmin
            # print ('a2',i, j, labelnew['eachfloor'][str(j)]['floor']['hf'], labelnew['eachfloor'][str(j+1)]['floor']['hf'], hspset[i][1]+hspmin)
            j+=2
        else:
            labelnew['eachfloor'][str(j)] = copy.deepcopy(label['eachfloor'][str(i)])
            # print ('bb',i, j, labelnew['eachfloor'][str(j)]['floor']['hf'])
            j+=1
    while hfset[-1]>hspset[-1][1]+hspmin and hfset[-1] > range_para['h_window'][0]:
        labelnew['eachfloor'][str(j)] = copy.deepcopy(label['eachfloor'][str(i)])
        labelnew['eachfloor'][str(j)]['floor']['hf'] = hspset[i][1]+hspmin
        hfset[-1] -= hspset[i][1]+hspmin
        # j+=1

    
    return labelnew

def decode(H, hfs, thisfloor, s=1):
   
    h_sp = H-np.sum(hfs)-thisfloor['floor']['hf']
    h_gap = h_sp + thisfloor['h_gap']
    # print ('nf',nf,h_sp)
    w_sp = thisfloor['w_bound']
    nv=-1
    nm=0
    p_component = []
    num_eachwindow = thisfloor['num_eachwindow']
    edge = thisfloor['edge']
    nwindow = thisfloor['nwindow']
    if thisfloor['num_vertices']>1:
        for n in range(nwindow):
            if nm<n<num_eachwindow[str(nv+1)]['sw']+nm:
                w_sp+=num_eachwindow[str(nv+1)]['ww']+edge['d_intra'][str(2*(nv+1))]
            elif n==num_eachwindow[str(nv+1)]['sw']+nm:
                nm+=num_eachwindow[str(nv+1)]['sw']
                # print ('aaaaaaaaa',nm,edge['d_inter'],num_eachwindow)
                try:
                    w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                except:
                    # print ('error', thisfloor)
                    sys.exit()
                nv+=1
            p_component.append(((w_sp/s,h_gap/s),num_eachwindow[str(nv+1)]['ww']/s,num_eachwindow[str(nv+1)]['hw']/s))
            
    else:
        for n in range(nwindow):
            w_sp = thisfloor['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
            p_component.append(((w_sp/s,h_gap/s),num_eachwindow['0']['ww']/s,num_eachwindow['0']['hw']/s)) 
    return p_component

def encode(rects):
    return inverse_facade.creadeachfloor(rects) # ne,edge,nw

def add_firstwindow(meansp, samefloor, W):
    
    wbmin = range_para['w_bound'][0]
    mm=len(samefloor)
    (x0,y0),w0,h0 = samefloor[0]
    if x0 >= range_para['w_bound'][0] + meansp + w0:
        samefloor.insert(0, ((x0-meansp-w0, y0),w0,h0))
    
    return samefloor
        

def adj_vert(pwindows, meansp, W):
    
    samefloor=sorted(pwindows, key=lambda component: component[0][0])
    mm=len(samefloor)
    
    newwindowset = [samefloor[0]]
    
    if mm>1:
        for j in range(1, mm):
            newwindowset.append(samefloor[j])
            (x1,y1),w1,h1=samefloor[j-1]
            (x2,y2),w2,h2=samefloor[j]
            if x2-x1-w1 >= 2*meansp + (w1+w2)/2:
                # newwindowset.insert(j, (((x1+x2)/2,y1),w1,h1))
                newwindowset.insert(j, ((x1+w1+meansp,y1),w1,h1))
                # print ('insert a window at the positon', j,  (((x1+x2)/2,y1),w1,h1))
            j+=1
            # print (j, mm, W+w2-x2, x2-x1-w1, 2*range_para['w_sp'][0] + (w1+w2)/2, )

        # if W-w2-x2 >= 2*(x2-x1-w1)+w2:#samefloor[0][0][0]:
            # newwindowset.insert(j+1, ((2*x2-x1-w1+w2,y1),w1,h1))
        if x2+2*w2+meansp <= W-range_para['w_sp'][0]:
            # newwindowset.insert(j+1, ((x2+meansp+2*w1,y1),w1,h1))
            # newwindowset.insert(j+1, ((2*x2-x1-w1+w2,y1),w1,h1))
            newwindowset.insert(j+1, ((x2+w2+meansp,y1),w1,h1))
            # print ('add a window at the end', len(newwindowset), len(pwindows))
    else:
        (x1,y1),w1,h1 = samefloor[0]
        if x1+meansp+2*w1 <= W-range_para['w_sp'][0]:
            newwindowset.insert(1, ((x1+w1+meansp,y1),w1,h1))
        # print ('add left windows')
        # if x1-meansp-2*w1 > range_para['w_sp'][0] and x1>=range_para['w_sp'][1]:
        #     newwindowset.insert(0, ((x1-meansp-w1,y1),w1,h1))

    if len(newwindowset) != len(samefloor):
        stat = 'c'
    else:
        stat = 'o'
    
    return newwindowset, stat

                

def mean_spacing(pwindows):
    samefloor=sorted(pwindows, key=lambda component: component[0][0])
    mm=len(samefloor)
    spset = []
    if mm > 1:
        for i in range(1,mm):
            (x1,y1),w1,h1=samefloor[i-1]
            (x2,y2),w2,h2=samefloor[i]
            spset.append(x2-x1-w1)
    else:
        spset.append(samefloor[0][1])
    
    meansp = np.min(spset)
    
    if mm>1:
        ww1 = samefloor[spset.index(meansp)][1]
        ww2 = samefloor[spset.index(meansp)+1][1]
        if meansp- (ww1 + ww2)  > 2*range_para['w_sp'][0]:
            meansp = (meansp-samefloor[spset.index(meansp)+1][1])/2
    
    # print ('meansp-----', spset,  meansp)
    return meansp
    

def window_eachfloor(i,label,para_set,s=1):
    
    p_component=[]
    num_floor=0
    h_sp=H
    eachfloor=para_set[label]['eachfloor']
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
                    try:
                        w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                    except:
                        # print ('error', eachfloor[str(i)])
                        sys.exit()
                    nv+=1
                p_component.append(((w_sp/s,h_gap/s),num_eachwindow[str(nv+1)]['ww']/s,num_eachwindow[str(nv+1)]['hw']/s))
                
        else:
            for n in range(nwindow):
                w_sp=eachfloor[str(i)]['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
                p_component.append(((w_sp/s,h_gap/s),num_eachwindow['0']['ww']/s,num_eachwindow['0']['hw']/s)) 
    return p_component

def createfacade(W, H, lab_info,side):
    
    def get_windows_edge(): # nw,ww,hw,wsp
        nes = {'0':{'sw':nw,'ww':ww,'hw':hw}}
        edge = {'d_intra':{'0':wsp}}
        return nes,edge
    
    if W>range_para['w_window'][0]+range_para['w_sp'][0] and H>range_para['h_floor'][0]:
        ww = round(np.random.uniform(range_para['w_window'][0], min(W-2*range_para['w_sp'][0],range_para['w_window'][1])),2)
        hw = round(np.random.uniform(range_para['h_window'][0], min(H-2*range_para['h_gap'][0], range_para['h_window'][1])),2)
        wsp = round(np.random.uniform(range_para['w_sp'][0], min((W-ww)/2,range_para['w_sp'][1])),2)
        if side != 'side':
            nw = int(np.floor((W-wsp)/(ww+wsp)))
            wb = (W-nw*(ww+wsp))/2
        else:
            nw = 1
            wb = (W-ww)/2
        h_gap = round(np.random.uniform(range_para['h_gap'][0], min((H-hw)/2, range_para['h_floor'][1]-hw, range_para['h_gap'][1])),2)
        hf = max(range_para['h_floor'][0], hw+h_gap)#round(np.random.uniform(range_para['h_floor'][0], min(hw+h_gap,range_para['h_floor'][1])),2)
        nf = int(np.floor(H/hf))
        lab_info['eachfloor']['0'] = {'floor': {'sf': nf,
             'wf': W,
             'hf': hf},
            'w_bound': wb,
            'num_vertices': 1,
            'nwindow': nw,
            'h_gap': 0.0,
            'mode': 'sysm',
            'num_eachwindow': {'0':{'sw':nw,'ww':ww,'hw':hw}},
            'edge': {'d_intra':{'0': wsp}, 'd_inter': {}}}
        # print (lab_info)
    return lab_info
        
        

def drawrect(imgpath, imgname, img_save, W, H, p_window_o_set, pwindows_set, s):
    imgpath = imgpath + '/' +imgname+'.jpg'
    print ('draw',imgpath)
    img = cv2.imread(imgpath)
    WI = int(W*s)
    HI = int(H*s)
    for p_window_o in p_window_o_set.values():
        if p_window_o!=[]:
            for (x,y),w,h in p_window_o:
                pt1 = (int(x*s),HI-int(y*s))
                pt2 = (int((x+w)*s),HI-int((y+h)*s))
                color = (0,0,255)
                cv2.rectangle(img, pt1, pt2, color,2)
    for label, pwindows in pwindows_set.items():
        if pwindows!=[]:
            if label == 'window' or label == 'Stairwell_window':
                for ((x,y),w,h) in pwindows:
                    pt1 = (int(x*s),HI-int((y)*s)) #绘制时需要注意y坐标的正确写入
                    pt2 = (int((x+w)*s),HI-int((y+h)*s))
                    color = (255,255,0)
                    cv2.rectangle(img, pt1, pt2, color,1)
            else:
                print ('label',label,pwindows)
                for ((x,y),w,h) in pwindows:
                    pt1 = (int(x*s), HI-int((y)*s)) #绘制时需要注意y坐标的正确写入
                    pt2 = (int((x+w)*s),HI-int((y+h)*s))
                    color = (255,0,255)
                    cv2.rectangle(img, pt1, pt2, color,2)
            filename = img_save+'/'+'_'+'facade_OPT_'+imgname+'.png'
    else:
        filename = img_save+'/'+'_'+'opt_facade_'+imgname+'.png'
    print (filename)
    cv2.imwrite(filename, img)

# def window_spacing(p_component, i, floors, s=1):
#     width_median = stat(p_component)
#     num_eachwindow=eachfloor[str(i)]['num_eachwindow']
#     edge=eachfloor[str(i)]['edge']

def cal_wwr(H, W, pwindows, s):
    totalarea = 0
    for (x, y),w,h in pwindows:
        totalarea

def adddoor(W, H, pwindows_set, s):
    #add a door 
    p_window = pwindows_set['window']
    p_Stairwell_window = pwindows_set['Stairwell_window']
    
    if len(p_Stairwell_window)>1: #如果存在梯间窗户，则每个梯间窗的一楼为门的位置
        door = []
        xdoor = []
        yset = []
        for (x,y),w,h in p_Stairwell_window:
            yset.append(y)
            if (x,w,h) not in xdoor:
                xdoor.append((x,w,h))
                
        for x,w,h in xdoor:
            wdoor = min(w*1.5, 2)
            hdoor = max(2.2,h*1.5) #设定门的高度正常为2.2最低
            xd = x+w/2-wdoor/2
            yd = 0.1
            if ((xd,yd),wdoor,hdoor) not in door:
                door.append(((xd,yd),wdoor,hdoor))
            ym=np.min(yset)
            
            if ((x,ym),w,h) in p_Stairwell_window and ym-hdoor<0:
                # print ('ym-hdoor',ym-hdoor, p_Stairwell_window)
                p_Stairwell_window.remove(((x,ym),w,h))
                
    else: #如果不存在梯间窗户，则当立面宽度较大时，（大于16米）取楼栋的中间位置为门，否则，取左侧为入户门，门的尺寸固定
        if W>16:
            door = [((W/2-1,0.1),2,2.8)]
        else:
            door = [((1.0,0.1),1.8,2.5)]
    
        repeatwindow = []
        for (x,y),w,h in p_window:
            for (x2,y2),w2,h2 in door:
                if abs(y-y2)<=h+range_para['h_gap'][0] and abs(x-x2)<=w+range_para['w_sp'][0]:
                    repeatwindow.append(((x,y),w,h))
        
        for  (x,y),w,h in repeatwindow:
            # print ('delwindow', (x,y),w,h)
            if ((x,y),w,h) in p_window:
                p_window.remove(((x,y),w,h))
    
    return p_window,p_Stairwell_window,door
    
def delwindows(W,H,pwindows_set,s):
    
    # p_window = pwindows_set['window']
    # p_Stairwell_window = pwindows_set['Stairwell_window']
    p_window,p_Stairwell_window,door = adddoor(W,H,pwindows_set,s)
    # print ('p_Stairwell_window',p_Stairwell_window)
    repeatwindow = []
    if p_Stairwell_window!=[]:
        for (x,y),w,h in p_window:
            for (x2,y2),w2,h2 in p_Stairwell_window:
                if abs(y-y2)<=h+range_para['h_gap'][0] and abs(x-x2)<=w+range_para['w_sp'][0]:
                    repeatwindow.append(((x,y),w,h))
        
        for  (x,y),w,h in repeatwindow:
            # print ('delwindow', (x,y),w,h)
            if ((x,y),w,h) in p_window:
                p_window.remove(((x,y),w,h))

    # print ('aaaaaaaaa',pwindows_set)
    paraset_window = inverse_facade.deducepara(H,p_window,'window',s=1,laststep = 'laststep') #H,p_window,label, s=1
    
    paraset_Stairwell_window = inverse_facade.deducepara(H,p_Stairwell_window,'p_Stairwell_window',s=1,laststep = 'laststep') #H,p_window,label, s=1
    
    paraset_door = inverse_facade.deducepara(H,door,'door',s=1) #H,p_window,label, s=1
    # print ('aaa',paraset_window)
    
    return paraset_window,paraset_Stairwell_window,paraset_door

# def optimallayout(lab_info):
    
    
    
    
    

def amend(image_path, semantic_path, s, side):
    p_window_sam, para_set_creat, iou = initpara_batch3.get_para(image_path, semantic_path, s, side)

    para_set = para_set_creat['para_set']
    
    W =  para_set_creat['W']
    H =  para_set_creat['H']
    
    # p_window_o = window_rect_complex(W,H,para_set,'window')
    p_window_o_set = {}
    p_window_new_set = {}
    spset = [] 
    wbset = []
    #-------------RUNING AMENDING PROGRAM-----------------------
    for label in list(para_set.keys()):
        lab_info = para_set[label]
        p_window_o = window_rect_complex(W,H,para_set,label)
        p_window_o_set[label] = p_window_o
        print ('keys', label)
        print ('iou', iou)
        if len(list(lab_info['eachfloor'].keys()))>0:
            for opt in range(10):
                # print ('******-------opt----------******',label)
                hfset = []
                hspset = []
                if  iou < 1.8:
                    for numfloor in list(lab_info['eachfloor'].keys()):
                        # print ('opt:', opt, ' ','numfloor:', numfloor, len(list(lab_info['eachfloor'].keys())))
                        sf,wf,hf = lab_info['eachfloor'][numfloor]['floor'].values()
                        # floor_windows = window_eachfloor(numfloor,label, para_set, s=1)
                        pwindows = decode(H, hfset, lab_info['eachfloor'][numfloor])
                        
                        hspset.append((H-np.sum(hfset)-pwindows[0][0][1]-pwindows[0][2],pwindows[0][2]))
                        hfset.append(hf)
                        if label == 'window' and side != 'side':
                            # print ('pwindows', pwindows[0])
                            wb = lab_info['eachfloor'][numfloor]['w_bound']
                            if opt == 0:
                                meansp = mean_spacing(pwindows)
                                wbset.append(pwindows[0][0][0])
                            else:
                                meansp = np.min([np.min(spset), mean_spacing(pwindows)])
                            
                            if wb > range_para['w_bound'][0]+pwindows[0][1]+meansp: #np.min(wbset)+pwindows[0][1]+meansp:
                                # print ('wb',wb, np.min(wbset), pwindows[0][1], meansp)
                                pwindows = add_firstwindow(meansp, pwindows, W)
                            # lab_info['eachfloor'][numfloor]['w_bound'] = pwindows[0][0][0]
                            # print ('pwindowssssssss', pwindows[0])
                            adjwindows,stat = adj_vert(pwindows, meansp, W)
                            # while stat == 'c':
                                # print ('---------------')
                            adjwindows,stat = adj_vert(adjwindows, meansp, W)
                            adjwindows = add_firstwindow(meansp, adjwindows, W)
                            lab_info['eachfloor'][numfloor]['w_bound'] = adjwindows[0][0][0]
                            # print ('adjwindows', len(adjwindows))
                            ne,edge,nw = encode(adjwindows)
                            # print ('ne,edge,nw', ne,edge,nw)
                            if lab_info['eachfloor'][str(numfloor)]['num_eachwindow'] != ne:
                                lab_info['eachfloor'][str(numfloor)]['num_eachwindow'] = ne
                                lab_info['eachfloor'][str(numfloor)]['edge'] = edge
                                lab_info['eachfloor'][str(numfloor)]['nwindow'] = nw
                                lab_info['eachfloor'][str(numfloor)]['num_vertices'] = len(list(ne.keys()))
                            spset.append(meansp)
                            wbset.append(pwindows[0][0][0])
                            t = (opt, numfloor)
                    # plot(t, W, H, para_set, p_window_o)
                    if  label == 'window' or label == 'Stairwell_window' and len(list(lab_info['eachfloor'].keys()))>0: #opt == 0 and 
                        # print ('hfset11', hfset, pwindows[0][2])
                        if H-np.sum(hfset)-pwindows[0][2] > range_para['h_window'][0] or any(np.array(hfset) >= range_para['h_floor'][1]):
                            hspset.append((H-np.sum(hfset)-pwindows[0][2],pwindows[0][2]))
                            hfset.append(H-np.sum(hfset)-1)
                            # if hfset[-1] >= range_para['h_floor'][0]:
                            labelnew = adj_floor(H, hfset, hspset, para_set[label])
                            para_set[label] = labelnew
                            lab_info = para_set[label]
                            # plot(t, W, H, para_set, p_window_o)
                        else:
                            break
                        
        else:
            if label == 'window':
                # print (side)
                para_set[label] = createfacade(W, H, lab_info,side)
                t = ('opt', 'blank')
                p_window_new_set[label] = window_rect_complex(W,H,para_set,label)
                return p_window_o_set, p_window_new_set, para_set_creat, W, H
                # plot(t, W, H, para_set, p_window_o)
        # if para_set['Stairwell_window']['eachfloor']!={}:

        p_window_new_set[label] = window_rect_complex(W,H,para_set,label)

            
    para_set_creat['para_set']['window']['eachfloor'],para_set_creat['para_set']['Stairwell_window']['eachfloor'],para_set_creat['para_set']['door']['eachfloor'] = delwindows(W,H,p_window_new_set,s)
    for label in list(para_set_creat['para_set'].keys()):
        p_window_new_set[label] = window_rect_complex(W,H,para_set_creat['para_set'],label)
    # break
    return p_window_o_set, p_window_new_set, para_set_creat, W, H

def optimalwindows(img_read,imgpath,relfile,para_adjust):
    # img_read = r'cesgld'
    # path_save = r'save_para'
    img_save = r'opt_6th'
    # relfile = 'relsmodel.pkl'
    if os.path.exists(img_save) is not True:
        os.mkdir(img_save)
    
    files = os.listdir(img_read)
    ct=0
    # rel = joblib.load(relfile)
    for mpn,value1 in relfile.items():
        img = value1['img']
        imgname = img.split('/')[-1].split('.')[0]
        scale = value1['scale']
        side = 'main'#value1['side']
        print (ct,'filename',imgname,scale,side)

        semantic_path = img_read + '/' + imgname + '.pkl'
        image_path = img_read + '/'  + imgname+'_splash.png'
    # # get the shape W,H of image
        if scale>0:
            p_window_o, p_window_new,para_set_creat, W, H = amend(image_path, semantic_path, scale,side)
            drawrect(imgpath, imgname, img_save, W, H, p_window_o, p_window_new, scale)
            # break
            joblib.dump(para_set_creat, para_adjust+'/' +imgname+'.pkl')
        
        ct+=1
        
        cal_iou(p_window_o['window'], p_window_new, scale)

    
if __name__ == '__main__':
    # img_read = r'result_5821'
    para_adjust = r'adjust_para2'
    # img_save = r'opt_1st\\'
    # files = os.listdir(img_read)
    # ct=0
    # rel = joblib.load('img_model.pkl')
    # for value1 in rel.values():
    #     img = value1['img']
    #     imgname = img.split('/')[-1].split('.')[0]
    #     scale = value1['scale']
    img_read = r'ces2'
    imgpath = r'appearence_rect2'
    imgname = 'GEOM_824792_0_tex_2245459'
    img_save = r'optimage'
    if os.path.exists(img_save) is not True:
            os.mkdir(img_save)
    scale = 13
    semantic_path = img_read + '/' + imgname + '.pkl'
    image_path = img_read + '/' + imgname+'_splash.png'
    # get the shape W,H of image
    p_window_o, p_window_new,para_set_creat, W, H = amend(image_path, semantic_path, scale,'main') #可以调整这里的side或main查看差别
    drawrect(imgpath, imgname, img_save, W, H, p_window_o, p_window_new, scale)
    # drawrect(imgpath, imgname, img_save, W, H, p_window_o, p_window_new, scale, mpn)
    plot((1,2), W, H, para_set_creat['para_set'], p_window_new['window'])
    joblib.dump(para_set_creat, para_adjust+'/'+imgname+'.pkl')
    joblib.dump(para_set_creat, img_save+'/'+imgname+'.pkl')
    cal_iou(p_window_o['window'], p_window_new, scale) 





















    
    