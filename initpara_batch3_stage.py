# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 09:52:38 2020

@author: wpc
"""

# import numpy as np
# import matplotlib.pyplot as plt 
# import copy
import inverse_facade_stage as inverse_facade
import os
# import sys
import joblib
# import skimage
# from PIL import Image
import cv2
# from skimage import io
# import json

def get_para(image_path, semantic_path, scale, side='main'):
    # image_path=r'D:\mask\Mask_RCNN-master\samples\facade\expr\\s_f_010_splash_20200729T172210.png'
    # para_path=r'D:\mask\Mask_RCNN-master\samples\facade\expr\\s_f_010.pkl'
    
    # get the shape W,H of image
    img=cv2.imread(image_path)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    H,W=gray.shape
    # x=joblib.load(semantic_path)
    x = joblib.load(semantic_path)
    p_window, para_set, iou = inverse_facade.getparaset(x,H,W,scale,side)
    para_set_create={'W':W/scale,'H':H/scale,'R':0.0,'para_set': para_set}

    
    return p_window, para_set_create, iou #,p_window

# para_all = {}

if  __name__ == "__main__":
    path_read = r'result_trondheim'
    path_save = r'save_para'
    files = os.listdir(path_read)
    ct=0
    rel = joblib.load('img_model.pkl')
    for value1 in rel.values():
        img = value1['img']
        imgname = img.split('/')[-1].split('.')[0]
        para_path = path_read + '/' + imgname + '.pkl'
        image_path = path_read + '/' + imgname+'_splash.png'
        scale = value1['scale']
        # try:
        para_set_create = get_para(image_path, para_path, scale)
        
        # para_all[imgname] = para_set_create
            
        joblib.dump(para_set_create, path_save+'/'+imgname+'.pkl')
        # except:
        #     continue
        
        print (ct,'filename',imgname)
        ct+=1
        
    # joblib.dump(para_all, path_save+'/'+'para_all'+'.pkl')
















