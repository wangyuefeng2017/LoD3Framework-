# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:59:39 2020

@author: wpc
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import draw
import readgml1


def order_points(modelpts,psts):
    rect = np.zeros((4, 2), dtype = "float32")
    newmp=[]
    # for i in range(1,len(modelpts)):
    #     x1,y1,z1 = modelpts[i-1]
    #     x2,y2,z2 = modelpts[i]
    #     x=(x2-x1)
    #     y=(y2-y1)
    #     z=(z2-z1)
    for x,y,z in modelpts:
        # newmp.append((x+y,z))
        newmp.append((np.sqrt(x*x+y*y),np.abs(z)))
    # print (newmp)
    newmp=np.array(newmp)
    
    s = newmp.sum(axis = 1)
    rect[3] = psts[np.argmin(s)]
    rect[1] = psts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(newmp, axis = 1)
    rect[2] = psts[np.argmin(diff)]
    rect[0] = psts[np.argmax(diff)]
    
    x1,y1,z1 = modelpts[np.argmax(diff)]
    x2,y2,z2 = modelpts[np.argmax(s)]
    w = np.sqrt((x2-x1)**2+(y2-y1)**2)
    h = abs(newmp[np.argmax(diff)][1] - newmp[np.argmin(diff)][1])
    # print (w,z2,z1,h)
    if w > 2 and h > 2:
        hw = h/w
    else:
        hw = 0
    
    return rect,hw,h
    

def four_point_transform(image, mpts, psts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect,hw,h = order_points(mpts, psts)
    (tl, tr, br, bl) = rect
    # rect = np.zeros((4, 2), dtype = "float32")
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    if hw != 0:
        maxHeight=int(maxWidth*hw)
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect,dst)#(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        scale = maxHeight/h
    else:
        warped = []
        scale = 0
    # print (M,(maxWidth, maxHeight)) 
    # return the warped image
    return warped,scale

def clipimage(image,vert,imname):
    
    roi_vert = np.asarray(vert)
    roi_vert = np.expand_dims(roi_vert, axis=0)
    # im = np.zeros(image.shape[:2], dtype = "uint8")
    # cv2.polylines(image, roi_vert, 1, 255)
    # cv2.fillPoly(im, roi_vert, 255)
    xset=[]
    yset=[]
    for x,y in vert:
        xset.append(x)
        yset.append(y)
    rr,cc=draw.polygon_perimeter(yset,xset)
    draw.set_color(image,[rr,cc],[255,0,0])
    
    return image
    
    
# from pyimagesearch.transform import four_point_transform
# import numpy as np
# import argparse
# import cv2
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image file")
# ap.add_argument("-c", "--coords",
#     help = "comma seperated list of source points")
# args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them

# image = cv2.imread(args["image"])
# pts = np.array(eval(args["coords"]), dtype = "float32")

if  __name__ == "__main__":
    imagepath=r'../citygml_textures/tex4000.jpg'
    # psto=np.array([(0.997868, 0.281073), 
    #                (0.645234, 0.995344), 
    #                (0.002454, 0.744746), 
    #                (0.373885, 0.004912), 
    #                (0.997868, 0.281073)])
    ring = '0.165066 0.999639 0.004767 0.865276 0.839491 0.003351 0.994034 0.132886 0.165066 0.999639'
    # mpts=np.array([(392578.799540847, 5816874.22092948, 58.1792922242878), 
    #                (392583.291561595, 5816872.05187734, 58.1792922242878),
    #                (392585.806510069, 5816877.09634229, 56.1817517140715), 
    #                (392578.799540847, 5816874.22092948, 58.1792922242878)])
    LinearRing = '392957.565269842 5817514.48862051 59.1983286557573 392958.652330213 5817516.59666674 59.1983286557573 392958.651813171 5817516.59590369 35.0299987792969 392957.5647528 5817514.48785747 35.0299987792969 392957.565269842 5817514.48862051 59.1983286557573'
    
    image = cv2.imread(imagepath)
    s = image.shape
    mpts=readgml1.transpos(LinearRing)
    psts = readgml1.transcoord(ring,s)
    imaged=clipimage(image,psts,imagepath)
    fig=plt.figure(dpi=120)
    ax = fig.add_subplot(221)
    plt.imshow(imaged)
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped,scale = four_point_transform(imaged,mpts, psts)
    
    # show the original and warped images
    # cv2.imshow("Original", image)
    # cv2.imshow("Warped", warped)
    # cv2.waitKey(0)
    
    # fig=plt.figure(dpi=120)
    ax = fig.add_subplot(222)
    plt.imshow(warped)
    imagedraw=readgml1.clipimage(image, psts)
    # plt.imshow(imagedraw)

# [(0.997868, 0.281073), (0.645234, 0.995344), (0.002454, 0.744746), (0.373885, 0.004912), (0.997868, 0.281073)]
# 0.997868 0.281073 0.645234 0.995344 0.002454 0.744746 0.373885 0.004912 0.997868 0.281073
# [(392578.799540847, 5816874.22092948, 58.1792922242878), (392583.291561595, 5816872.05187734, 58.1792922242878),
# (392585.806510069, 5816877.09634229, 56.1817517140715), (392578.799540847, 5816874.22092948, 58.1792922242878)]















