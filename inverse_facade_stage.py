# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 01:01:44 2020

@author: wyf-laptop
"""

# import numpy as np
import matplotlib.pyplot as plt 
import copy
# import os
import joblib
# import skimage
from PIL import Image
import numpy as np
import cv2

range_para={'h_floor':(2.2,5.5),
            'h_first':(3.0,5),
            'h_roof':(0.1,1),
            'h_gap':(0.1,1.0),
            'h_fgap':(0.0,2.5),
            'h_window':(0.5,2.8),
            'w_window':(0.4,3.8),
            'w_sp':(0.2,2.0),
            'w_bound':(0.5,3.5)}



def cal_iou(rectangles_from_segmentation, rectangles_after_regularization,s):
    '''
        计算交并比
        输入：rectangles_from_segmentation为sam+yolo得到的窗户矩形，形式为[((x,y),w,h)]
              rectangles_after_regularization为生成的窗户矩形，形式为{‘window’:[],'stairwell_window':[]}
        输出：交并比的值（0-1）

    '''
    def are_rectangles_intersecting(rect1, rect2):
        (x1_1, y1_1), w1, h1 = rect1
        (x1_2, y1_2), w2, h2 = rect2

        # 计算两个矩形的边界
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # 检查是否有任何一个矩形在另一个矩形的左侧、右侧、上方或下方
        if x1_1 > x2_2*s or x1_2*s > x2_1 or y1_1 > y2_2*s or y1_2*s > y2_1:
            return False
        else: 
            return True
        
    def calculate_iou(rect1, rect2):
        # rect: ((x, y), w, h)
        (x1, y1), w1, h1 = rect1
        (x2, y2), w2, h2 = rect2
    
        # 计算交集区域
        if are_rectangles_intersecting(rect1, rect2):
            x_intersection = max(0, min(x1 + w1, int(x2*s) + int(w2*s)) - max(x1, int(x2*s)))
            y_intersection = max(0, min(y1 + h1, int(y2*s) + int(h2*s)) - max(y1, int(y2*s)))
            intersection_area = x_intersection * y_intersection
        
            # 计算并集区域
            if intersection_area != 0:
                union_area = w1 * h1 + int(w2*s) * int(h2*s) - intersection_area
            else:
                union_area = 0
        
            # 计算IOU
            iou = intersection_area / union_area if union_area > 0 else 0
        else:
            iou = 0
    
        return iou
    
    def calculate_average_iou(rectangles1, rectangles2):
        total_iou = 0
        total_pairs = 0
        for rect1 in rectangles1:
            for rect2set in rectangles2.values():
                for rect2 in rect2set:
                    iou = calculate_iou(rect1, rect2)
                    # print ('iou',iou)
                    if iou>0:
                        total_iou += iou
            total_pairs += 1
    
        average_iou = total_iou / total_pairs if total_pairs > 0 else 0
        
        print ('total_pairs',total_iou, total_pairs)
        return average_iou
    
    # # 示例用法
    # rectangles_from_segmentation = [((x1, y1), w1, h1), ((x2, y2), w2, h2), ...]  # 实例分割结果
    # rectangles_after_regularization = [((x3, y3), w3, h3), ((x4, y4), w4, h4), ...]  # 规则化后的结果
    
        # 计算平均IoU
    average_iou = calculate_average_iou(rectangles_from_segmentation, rectangles_after_regularization)
    
    print("Average IoU:", average_iou)
    return average_iou


def findy(array_list, target_y,s):
    '''
        查找确定窗的y后对应的矩形窗户索引
        输入：array_list为s窗户矩形的列表，形式为[((x,y),w,h)]
              targetx_y为需要检索的矩形中的y坐标
              s为图像与模型立面的尺寸比
        输出：candidate_i检索出来的矩形位置，这里是为了通过y坐标找到对应的矩形，从而确定窗所在的列数和窗户的高度和宽度

    '''
    candidate_i = []
    for i, ((x, y), w, h) in enumerate(array_list):
       if abs(y - target_y) < range_para['h_window'][0]*s+range_para['h_gap'][0]*s: #查找确定y的前提下梯间窗对应的窗户矩形
           candidate_i.append(i)
    return candidate_i
    # return -1

def find_coldis(arr, col):
    '''
        查找每一行所对应的索引，以便找出该行存在的窗户个数
        输入：arr为行的统计列表，形式为[[col,num]]
              col为需要检索的行的坐标，也就是窗户矩形通过聚类后得到的y坐标
        输出：i检索出来的行的位置

    '''
    for i, (cols,num) in enumerate(arr):
        if col == cols:
            return i
    return -1

def getrow_col(p_window,H,W,s=1,side='main'):
    '''
        对实例分割的结果进行规则化，将存在较小差异的矩形的x坐标和y坐标按照一定的阈值进行聚类和对齐
        输入：p_window为窗户矩形
              s为图像与模型立面的尺寸比
              side为立面的位置，分为main和side
        输出：pwindow_new, Stairwell_window分别是规则化后得到的窗户矩形和梯间窗矩形列表

    '''
    pwindownew = []
    row = []
    col = []
    wset = []
    hset = []
    for (x,y),w,h in p_window:
        # print (w/s,h/s)
        if w/s<range_para['w_window'][0] or w/s>range_para['w_window'][1] or h/s<range_para['h_window'][0] or h/s>range_para['h_window'][1]:
            continue
        else:
            pwindownew.append(((x,y),w,h))
            row.append(x) #分别将每个窗户矩形的行、列、宽度、高度存放在不同的列表中，目的是为了得到最小的行和列的值
            col.append(y)
            wset.append(w) 
            hset.append(h)

    # return row,col,wset,hset
    if row!=[] and col!=[]:
        wmedian = np.median(wset)
        hmedian = np.median(hset)
        
        row2=sorted(row) # 对列表进行排序，按照从小到大的顺序依次处理，从而能够通过遍历查找相邻的行或列是否能够进行聚类
        # print ('row',row2)
        col2=sorted(col)
        # print ('col2',col2)
        slectrow = row2[0]
        slectcol = col2[0]
        rowdis = []
        coldis = []
        
        wdif = [wmedian]
        nearrow = 1 # nearrow是满足聚类条件的窗户的个数，当该数字大于一个阈值时，我们认为是有效的，否则认为对应的列为噪声，从而排除
        nearrowset = []
        eachrow = []
        eachwidth = []
        rowwidth = []
        # print ('rr',wmedian+range_para['w_sp'][0]*s,range_para['w_window'][0]*s+range_para['w_sp'][0]*s)
        
        #--------------------聚类水平方向x轴-----------------------------------
        sorted_pwindownew = sorted(pwindownew, key=lambda building: building[0][0])
        if slectrow > range_para['w_sp'][0]*s:
            start = 0
        else:
            start = 1
        for i in range(start, len(sorted_pwindownew)):
        
            rdis = abs(sorted_pwindownew[i][0][0]-slectrow)
            # print (rdis,sorted_pwindownew[i][0][0],slectrow)
            '''
            #阈值1：rdis是用于确定相邻的两列能否合并，range_para['w_window'][0]*s+range_para['w_sp'][0]*s为最小的窗户宽度和窗户间隔。
            #当满足条件时，我们将这两个列进行合并，并使nearrow的个数加一，并将该列加入列表eachrow，该窗户的宽度加入列表eachwidth。
            #对竖直方向上也是同一的操作
            '''
            if rdis < range_para['w_window'][0]*s+range_para['w_sp'][0]*s: 
                nearrow+=1
                eachrow.append(sorted_pwindownew[i][0][0])
                eachwidth.append(sorted_pwindownew[i][1])
            else:
                if len(eachrow)>0:
                    # averagerow = sum(eachrow)/len(eachrow)
                    averagerow = np.median(eachrow)#sum(eachrow)/len(eachrow)
                    rowdis.append((int(averagerow),nearrow))
                else:
                    averagerow = slectrow
                    if rdis-wdif-range_para['w_sp'][0]*s>0:
                        rowdis.append((int(averagerow),nearrow))
                    
                # rowdis.append((int(averagerow),nearrow))
                slectrow = sorted_pwindownew[i][0][0]
                
                if len(eachwidth)>0:
                    # averagewidth = sum(eachwidth)/len(eachwidth)
                    averagewidth = np.median(eachwidth)
                    rowwidth.append(averagewidth)     
                else:
                    averagewidth = sorted_pwindownew[i][1]
                    if rdis-wdif-range_para['w_sp'][0]*s>0:
                        rowwidth.append(averagewidth)     
                # rowwidth.append(averagewidth)     
                eachwidth = []
                eachrow = []
                nearrow = 1

            
        #增加最右侧的窗户
        rowdis.append((int(slectrow),nearrow))
        if len(eachwidth)>0:
            averagewidth = np.median(eachwidth)#sum(eachwidth)/len(eachwidth)
        else:
            averagewidth = sorted_pwindownew[i][1]
        rowwidth.append(averagewidth)    
        
        
        #去除水平方向的噪声点
        # 出现次数少于x的被认为是假的，需要排除，由于每个立面上窗户数量不一致，因此需要通过估计的方法计算每层和每列窗户的最小数量
        visibleww = max(col)-min(col) #这里用到的立面宽度不是图像的宽度，而是我们分割结果中矩形的最小的y和最大的y。
        hor_win_num = int((visibleww/s)/(range_para['w_window'][1]+range_para['w_sp'][1])) #立面的宽度除以窗户的最大宽度和间隔的最大宽度
        rowdisclean = []
        widthclean = []
        for i in range(len(rowdis)):
            sr,numw = rowdis[i]
            if side=='side': # 最少出现的个数，当为side时设为>0，当为main时需>1
                minnum = 0
            else:
                minnum = 1
            if numw > max(minnum,hor_win_num/2-2):  #每列都存在2个未检测的窗户时，用估计的最小数量/2再减去2，以保证鲁棒
                rowdisclean.append(sr)
                widthclean.append(rowwidth[i])
        
        print ('dd',hor_win_num/2-2,rowdis,rowdisclean,rowwidth, widthclean)
                
                
        #--------------------聚类竖直方向y轴-----------------------------------
        nearcol = 1
        nearcolset = []
        eachcol = []
        eachheigt = []
        colheight = []
        
        sorted_pwindownew2 = sorted(pwindownew, key=lambda building: building[0][1])
        # print ('sorted_pwindownew2',sorted_pwindownew2,slectcol,min(hset))
        if slectcol > H - range_para['h_gap'][0]*s:
            start = 0
        else:
            start = 1
        for i in range(start, len(sorted_pwindownew2)):
            # cdis = abs(col2[i] - slectcol)
            cdis = abs(sorted_pwindownew2[i][0][1] - slectcol)
            # print ('cdis',cdis,slectcol,min(hset))
            if cdis < range_para['h_window'][0]*s+range_para['h_gap'][0]*s: #rdis < wmedian+range_para['w_sp'][0]*s and 
            # if cdis < min(hset)+range_para['h_gap'][0]*s:
                nearcol+=1
                eachcol.append(sorted_pwindownew2[i][0][1])
                eachheigt.append(sorted_pwindownew2[i][2])
            else:
                if len(eachcol)>0:
                    # averagerow = sum(eachrow)/len(eachrow)
                    averagecol = np.median(eachcol)#sum(eachrow)/len(eachrow)
                else:
                    averagecol = slectcol
                coldis.append((int(averagecol),nearcol))
                slectcol = sorted_pwindownew2[i][0][1]
                
                if len(eachheigt)>0:
                    # averageheight = sum(eachheigt)/len(eachheigt)
                    averageheight = np.median(eachheigt)
                else:
                    averageheight = hmedian
                    
                colheight.append(averageheight)     
                eachheigt = []
                eachcol = []
                nearcol = 1
                
        #增加最下层的窗户
        coldis.append((int(slectcol),nearcol))
        # print ('slectcol',slectcol)
        if len(eachheigt)>0:
            averageheight = sum(eachheigt)/len(eachheigt)
        else:
            averageheight = sorted_pwindownew2[i][2]#hmedian
        colheight.append(averageheight)    
        
        #去除竖直方向的噪声点
        visiblehw = max(row)-min(row)
        ver_win_num = int((visiblehw/s)/(range_para['h_window'][1]+range_para['h_gap'][1])) #立面的宽度除以窗户的最大宽度和间隔的最大宽度
        
        coldisclean = []
        heightclean = []
        for i in range(len(coldis)):
            sr,numw = coldis[i]
            # 出现次数少于3的被认为是假的，需要排除(需要考虑每层楼窗户的数量)
            # num = max(0,int(len(rowdisclean)/2)-1)
            if side=='side': # 最少出现的个数，当为side时设为>0，当为main时需>1
                minnum = 0
            else:
                minnum = 1
            if numw > max(minnum,ver_win_num/2-2): #0:#0:#
                coldisclean.append(sr)
                heightclean.append(colheight[i])
                
        print ('dd2',ver_win_num/2-2,coldis,coldisclean, colheight, heightclean)
                
        #---------------------查找是否存在梯间窗-------------- 
        diff_col = []
        row_Stairwell_window = []
        col_Stairwell_window = []
        width_Stairwell_window = []
        height_Stairwell_window = []
        
        '''
        #最底层楼层由于遭受遮挡的可能比较大，因此不参与到梯间窗的判定中，
        但是这个可能存在问题，后续调整的话可以尝试将len(coldisclean)-1改为len(coldisclean)
        '''
        
        for i in range(1, len(coldisclean)): 
            if coldisclean[i]-coldisclean[i-1] < max(heightclean) + range_para['h_gap'][0]*s: #如果竖直方向上相邻两个楼层间隔小于窗户间隔，则可能存在梯间窗
                num1 = coldis[find_coldis(coldis, coldisclean[i])][1]
                num2 = coldis[find_coldis(coldis, coldisclean[i-1])][1]
                # print ('stairwindow',num1,num2,coldisclean[i],coldisclean[i-1], max(heightclean))
                if num1 <= num2 and coldisclean[i] not in col_Stairwell_window: # 一般而言，梯间窗的数量要少于正常的窗户
                    col_Stairwell_window.append(coldisclean[i])
                    candidate_i = findy(sorted_pwindownew2, coldisclean[i],s) # 查找确定梯间窗的y后对应的矩形窗户索引
                    for stair_x in candidate_i:
                        stairerow = sorted_pwindownew2[stair_x][0][0]
                        if stairerow not in row_Stairwell_window:
                            row_Stairwell_window.append(stairerow)
                            width_Stairwell_window.append(sorted_pwindownew2[stair_x][1])
                            height_Stairwell_window.append(sorted_pwindownew2[stair_x][2])
                    # print ('stair',row_Stairwell_window)
                elif num1 > num2 and coldisclean[i-1] not in col_Stairwell_window:
                    col_Stairwell_window.append(coldisclean[i-1])

                    candidate_i = findy(sorted_pwindownew2, coldisclean[i-1],s) 
                    for stair_x in candidate_i:
                        stairerow = sorted_pwindownew2[stair_x][0][0]
                        if stairerow not in row_Stairwell_window:
                            row_Stairwell_window.append(stairerow)
                            width_Stairwell_window.append(sorted_pwindownew2[stair_x][1])
                            height_Stairwell_window.append(sorted_pwindownew2[stair_x][2])
                    # print ('stair2',row_Stairwell_window)
                    # row_Stairwell_window = findy(sorted_pwindownew2, coldisclean[i-1])

        if row_Stairwell_window!=[]:
            for rowi in row_Stairwell_window:
                for rowc in rowdisclean:
                    if abs(rowi-rowc) < range_para['w_window'][0]*s+range_para['w_sp'][0]*s:
                        rowindex = rowdisclean.index(rowc)
                        rowdisclean.remove(rowc)
                        del widthclean[rowindex]
                        # print ('rowindex',rowindex,rowdisclean,widthclean)

            
            for coli in col_Stairwell_window:
                for colc in coldisclean:
                    if abs(coli-colc) < range_para['h_window'][0]*s+range_para['h_gap'][0]*s:
                        colindex = coldisclean.index(colc)
                        coldisclean.remove(colc)
                        del heightclean[colindex]


            sortrowstair = sorted(row_Stairwell_window,reverse=True)
            
            rowstairclean = [sortrowstair[0]]
            for j in range(1,len(sortrowstair)):
                srowdis = abs(sortrowstair[j]-rowstairclean[-1])
                if srowdis > range_para['h_gap'][1]*s:
                    rowstairclean.append(sortrowstair[j])
            
            sortcolstair = sorted(col_Stairwell_window,reverse=True)
            
            colstairclean = [sortcolstair[0]]
            for j in range(1,len(sortcolstair)):
                scoldis = abs(sortcolstair[j]-colstairclean[-1])
                if scoldis > range_para['w_sp'][1]*s:
                    colstairclean.append(sortcolstair[j])
            
        print ('coldis',coldis,colheight,coldisclean, heightclean,rowdisclean,widthclean)
        # print ('stairwindowset', col_Stairwell_window, row_Stairwell_window, rowstairclean)

        pwindow_new = []
        Stairwell_window = []
        ctx = 0
        
        for x in rowdisclean:
            cty = 0
            for y in coldisclean:
                pwindow_new.append(((x,y),widthclean[ctx],heightclean[cty]))
                cty+=1
            ctx+=1
            
        if row_Stairwell_window != []:
            for x in rowstairclean:
                for y in colstairclean:
                    Stairwell_window.append(((x,y),width_Stairwell_window[0],height_Stairwell_window[0]))

        # pwindow_new = delwrong(pwindow_new)
        # Stairwell_window = delwrong(Stairwell_window)
        pwindow_new = check_and_remove_overlapping(pwindow_new)
        Stairwell_window = check_and_remove_overlapping(Stairwell_window)
        # print (Stairwell_window)
        return pwindow_new, Stairwell_window
    else:
        return [],[]

def is_overlap(rect1, rect2):
    # 判断两个矩形是否重叠
    (x1, y1), w1, h1 = rect1
    (x2, y2), w2, h2 = rect2
    return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)

def check_and_remove_overlapping(rectangles): #删除重叠的矩形,对不满足矩形之间不相交、距离太近等约束条件的窗户矩形进行删除，维持生成结果的拓扑正确。
    n = len(rectangles)
    to_remove = set()

    for i in range(n):
        for j in range(i + 1, n):
            if is_overlap(rectangles[i], rectangles[j]):
                # 标记要删除的矩形索引
                if rectangles[i][1] < rectangles[j][1]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    # 删除宽度较小的矩形
    rectangles = [rect for i, rect in enumerate(rectangles) if i not in to_remove]

    return rectangles


def getsw(cluster,s=1):
    '''
    当多个窗户之间具有间距和形状的差异小于阈值时，我们认为它们属于一个簇，在该簇中，
    我们进行一个规则化，将他们的间隔和形状调整为一致。
    
    输入：cluster为窗户矩形的列表。
    输出：sw,ww,dintra分别为窗户的个数、窗户的宽度，和窗户之间的间隔，
    当存在大于2个窗户时，间隔为((xa-xb)/s-(sw-1)*ww)/(sw-1)，仅有一个窗户时，间隔值设为0
    '''
    sw = len(cluster)
    if sw>1:
        (xa,ya),wa,ha=cluster[-1]
        (xb,yb),wb,hb=cluster[0]
        ww = (wa+wb)/2/s
        dintra = ((xa-xb)/s-(sw-1)*ww)/(sw-1)
    else:
        sw = 1
        ww = cluster[0][1]/s
        dintra = 0
    return sw,ww,dintra
      
def creadeachfloor(samefloor,s=1):
    '''
    当找到一个楼层中所有的窗户时，我们需要把这个楼层用一个拓扑图描述，具体采用参数表示。也就是论文里面的底层拓扑图。
    
    '''
    samefloor=sorted(samefloor, key=lambda component: component[0][0])
    # print ('pw',    samefloor)
    cluster=[samefloor[0]]
    num_eachwindow={}
    intra={}
    inter={}
    k=1
    '''
    定义一个窗户之间宽度的差异阈值，dif_width = 0.5
    '''
    dif_width = 0.5 #两个相邻的窗户之间如果宽度只差小于阈值0.5的话，认为这两个窗户应该是宽度一致的，需要进行宽度合并
    mm=len(samefloor)
    nw=0
    hm=0
    # yset=[]
    # for i in range(mm):
    #     yset.append(mm[i][0][1])
    # ymean=np.mean(yset)
    # print (samefloor)
    '''
    以下的if条件语句均用于判断如何识别多个窗户是否属于一个簇，该过程可以参考博士论文中的图4-5.
    '''
    if mm>2:
        j=1
        while j<=mm:
            # print (m,i,mm,j,samefloor[j])
            # print (num_eachwindow)
            if j<mm-1:
                (x1,y1),w1,h1=samefloor[j-1]
                (x2,y2),w2,h2=samefloor[j]
                (x3,y3),w3,h3=samefloor[j+1]
                hm=(h1+h2+h3)/3 #np.median([h1,h2,h3,hm])#
                d10=x2-x1-w1
                d11=x3-x2-w2
                if abs(d10-d11)/s<=0.5 and abs(w1-w2)/s<=dif_width and abs(w2-w3)/s<=dif_width:
                    cluster.append(samefloor[j])
                    # print ('1',cluster,j)
                    j+=1
 
                elif abs(w1-w2)/s<=dif_width and abs(w2-w3)/s>dif_width:
                    cluster.append(samefloor[j])
                    # print ('2',cluster,j)
                    sw,ww,dintra = getsw(cluster,s)
                    # print (sw,ww,dintra)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    intra[str(nw*2)]=dintra
                    inter[str(nw*2+1)]=(x3-x2-w2)/s
                    nw+=1
                    j+=2
                    cluster=[samefloor[j-1]]
                    
                elif (d11-d10)/s>0.5 and abs(w1-w2)/s<=dif_width: # and abs(w2-w3)<=(w2+w3)/10:
                    cluster.append(samefloor[j])
                    # print ('3',cluster,j)
                    sw,ww,dintra = getsw(cluster,s)
                    # print (d11,d10,sw,ww,dintra)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    intra[str(nw*2)]=dintra
                    inter[str(nw*2+1)]=(x3-x2-w2)/s
                    nw+=1
                    j+=2
                    cluster=[samefloor[j-1]]
                elif (d10-d11)/s>0.5 and abs(w1-w2)/s<=dif_width and abs(w2-w3)/s<=1:
                    # cluster.append(samefloor[j])
                    # print ('4',cluster,j)
                    sw,ww,dintra = getsw(cluster,s)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    # print ('bb',sw,ww,dintra)
                    if len(cluster)==1:
                        intra[str(nw*2)]=0
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    else:
                        intra[str(nw*2)]=dintra#(abs(cluster[0][0][0]-cluster[1][0][0])-cluster[1][1])/s
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    nw+=1
                    j+=1
                    cluster=[samefloor[j-1]]
                elif abs(w1-w2)/s>dif_width:
                    # print ('5',cluster,j)
                    sw,ww,dintra = getsw(cluster,s)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    if len(cluster)==1:
                        intra[str(nw*2)]=0
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    else:
                        intra[str(nw*2)]=dintra#(abs(cluster[0][0][0]-cluster[1][0][0])-cluster[1][1])/s
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    nw+=1
                    j+=1
                    cluster=[samefloor[j-1]]
                else:
                    # print ('else',samefloor[j-1],samefloor[j],samefloor[j+1])
                    sw,ww,dintra = getsw(cluster,s)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    if len(cluster)==1:
                        intra[str(nw*2)]=0
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    else:
                        intra[str(nw*2)]=dintra#(abs(cluster[0][0][0]-cluster[1][0][0])-cluster[1][1])/s
                        inter[str(nw*2+1)]=(x2-x1-w1)/s
                    nw+=1
                    j+=1
                    cluster=[samefloor[j-1]]
            elif j==mm-1:
                (x1,y1),w1,h1=samefloor[j-2]
                (x2,y2),w2,h2=samefloor[j-1]
                (x3,y3),w3,h3=samefloor[j]
                hm=(h1+h2+h3)/3
                d10=x2-x1
                d11=x3-x2
                if len(cluster)>1:
                    cluster.append(samefloor[j])
                    # print ('6',cluster,j)
                    sw,ww,dintra = getsw(cluster,s)
                    num_eachwindow[str(nw)]={'sw':sw,'ww':ww,'hw':hm/s}
                    intra[str(nw*2)]=dintra#(x2-x1-w1)/s
                    if nw==0:
                        inter={}
                    
                else:

                    if abs(w1-w2)/s<=dif_width and abs(w2-w3)/s<=dif_width and (d10-d11)/s>=0.5:
                        cluster.append(samefloor[j])
                        # print ('72',cluster,j)
                        num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w3/s,'hw':hm/s}
                        intra[str(nw*2)]=(x3-x2-w3)/s
                    elif abs(w1-w2)/s>dif_width and abs(w2-w3)/s<=dif_width:
                        cluster.append(samefloor[j])
                        # print ('72',cluster,j)
                        num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w3/s,'hw':hm/s}
                        intra[str(nw*2)]=(x3-x2-w3)/s
                        
                    elif abs(w2-w3)/s>dif_width or (d11-d10)/s>0.5:
                        # print ('8',cluster,j)
                        num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w2/s,'hw':hm/s}
                        intra[str(nw*2)]=0
                        inter[str(nw*2+1)]=(x3-x2-w2)/s
                        cluster=[samefloor[j]]
                        # print ('9',cluster,j)
                        nw+=1
                        num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w3/s,'hw':hm/s}
                        intra[str(nw*2)]=0
                    else:
                        cluster.append(samefloor[j])
                        # print ('11',cluster,j)
                        num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w3/s,'hw':hm/s}
                        intra[str(nw*2)]=(x3-x2-w3)/s
                break
            elif j==mm:
                (x1,y1),w1,h1=samefloor[-1]
                num_eachwindow[str(nw)]={'sw':len(cluster),'ww':w1/s,'hw':hm/s}
                intra[str(nw*2)]=0
                # print ('break')
                break

    elif mm==1:
        (x1,y1),w1,h1=samefloor[0]
        num_eachwindow[str(nw)]={'sw':1,'ww':w1/s,'hw':h1/s}
        intra[str(nw*2)]=0
        inter={}
        # edge={'d_intra':intra, 'd_inter':inter}
    else:
        (x1,y1),w1,h1=samefloor[0]
        (x2,y2),w2,h2=samefloor[1]
        if abs(w1-w2)/s<=dif_width:
            num_eachwindow[str(nw)]={'sw':2,'ww':(w1+w2)/2/s,'hw':h1/s}
            intra[str(nw*2)]=(x2-x1-w1)/s
            inter={}
            # edge={'d_intra':intra, 'd_inter':inter}
        else:
            num_eachwindow[str(nw)]={'sw':1,'ww':w1/s,'hw':h1/s}
            num_eachwindow[str(nw+1)]={'sw':1,'ww':w2/s,'hw':h2/s}
            intra[str(nw*2)]=0
            intra[str((nw+1)*2)]=0
            inter[str(nw*2+1)]=(x2-x1-w1)/s
            
    edge={'d_intra':intra, 'd_inter':inter}
    # print (num_eachwindow)
    lenwindows = len(samefloor)
    return num_eachwindow,edge,lenwindows

def createfloor(hv,samefloor):
    '''
    用于求解楼层的高度和宽度
    hv表示剩余可分配的高度值，samefloor为属于一个楼层的窗户矩形的集合。

    '''
    # h_gap=0 #default
    # p_wy=sorted(samefloor, key=lambda component1: component1[0][0],reverse=True)
    
    wf = samefloor[-1][0][0]
    hwset=[]
    for i in range(len(samefloor)):
        hwset.append(samefloor[i][0][1])
    
    hwmed = np.median(hwset)
    hf=hv-min(hwset)
    
    return wf,hf

def rightwindow(p_window_o,H,W,s,side):
    '''
    对实例分割结果的预处理，对应于方法getrow_col

    Parameters
    ----------
    p_window_o : 实例分割的窗户矩形.
    s : 图像和立面的尺寸比.
    side : 立面朝向  .

    Returns
    -------
    p_windowset : 
        规则化后的窗户.

    '''
    if len(p_window_o)!=0:
        p_windowset = getrow_col(p_window_o, H,W,s, side)
    else:
        p_windowset=[[],[]]
    return p_windowset
    
    
    
    


def deducepara(H,p_window,label, s=1, laststep=None):
    '''
    用于推断墙面布局拓扑图模型的主方法，根据输入的实例分割结果，得到一个初始状态的布局图参数集

    Parameters
    ----------
    H : 立面图像的高度、
    p_window : TYPE
        实例分割结果.
    label : TYPE
        立面元素类型，为窗户、梯间窗、门.
    s : TYPE, optional
        图像和模型立面的尺寸比. The default is 1.
    laststep : TYPE, optional
        修改后未使用. The default is None.

    Returns
    -------
    None.

    '''
    
    if len(p_window)==0:
        eachfloor={}
        return eachfloor
    if laststep:
        p_window = check_and_remove_overlapping(p_window)
    p_wy=sorted(p_window, key=lambda component: component[0][1],reverse=True) #对窗户按照y坐标进行从大到小的排序
    if len(p_wy)==0:
        eachfloor={}
        return eachfloor
    
    
    eachfloor={}
    nf=0
    mode='sysm'
    # num_eachwindow={}
    h_roof=range_para['h_roof'][0] #default
    h_gap= range_para['h_gap'][0] #default
    # H=p_window[0][0][1]+p_window[0][2]+h_roof
    hv=H
    m=len(p_wy)
    samefloor=[p_wy[0]]
    # diff_floor=[p_wy[0][1]]
    if len(p_wy) > 1:
        for i in range(1,m):
            # print (h_gap)
            (x1,y1),w1,h1=p_wy[i-1]
            (x2,y2),w2,h2=p_wy[i]
            if abs(y1-y2)<=0.5:
            # if abs((y1+h1)-(y2+h2))/s<=2 and abs(y1-y2)/s<=2:
                samefloor.append(p_wy[i])

            else:

                samefloor=sorted(samefloor, key=lambda component: component[0][0])
                # print (samefloor)
                num_eachwindow,edge,lenwindows=creadeachfloor(samefloor,s)
                wf,hf=createfloor(hv,samefloor)
                
                if hf>(2.8*s):  #2meter
                    eachfloor[str(nf)]={'floor':{'sf':1,'wf':wf/s,'hf':(hf+h_gap)/s},'w_bound':samefloor[0][0][0]/s,'num_vertices':len(num_eachwindow),
                                        'nwindow':lenwindows,'h_gap':h_gap/s,'mode':mode,'num_eachwindow':num_eachwindow,
                                        'edge':edge}
                else:
                    eachfloor[str(nf)]={'floor':{'sf':1,'wf':wf/s,'hf':(hf+h_gap)/s},'w_bound':samefloor[0][0][0]/s,'num_vertices':len(num_eachwindow),
                                        'nwindow':lenwindows,'h_gap':h_gap/s,'mode':mode,'num_eachwindow':num_eachwindow,
                                        'edge':edge}

                nf+=1
                hv=y1-h_gap
                samefloor=[p_wy[i]]

    else:
        (x1,y1),w1,h1=p_wy[0]
    samefloor=sorted(samefloor, key=lambda component: component[0][0])

    num_eachwindow,edge,lenwindows=creadeachfloor(samefloor,s)
    wf,hf=createfloor(hv,samefloor)
    eachfloor[str(nf)]={'floor':{'sf':1,'wf':wf/s,'hf':(hf+h_gap)/s},'w_bound':samefloor[0][0][0]/s,'num_vertices':len(num_eachwindow),
                    'nwindow':lenwindows,'h_gap':h_gap/s,'mode':mode,'num_eachwindow':num_eachwindow,
                    'edge':edge}
    
    # print (h_gap)
    if m==1:
        num_eachwindow,edge,lenwindows=creadeachfloor(samefloor,s)
        wf,hf=createfloor(hv,samefloor)
        eachfloor[str(nf)]={'floor':{'sf':1,'wf':wf/s,'hf':hf/s},'w_bound':samefloor[0][0][0]/s,'num_vertices':len(num_eachwindow),
                        'nwindow':lenwindows,'h_gap':h_gap/s,'mode':mode,'num_eachwindow':num_eachwindow,
                        'edge':edge}

    return eachfloor

def window_rect_complex(W,H,para_set,label): # label--> 'window','balcony','door'......
    '''
    根据推断出来的参数集生成语义墙面模型，表现形式为窗户矩形
    输入：W,H分别为模型立面的宽度和高度，para_set为参数集,label为立面上的元素
    输出：为窗户矩形
    '''
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
                        w_sp+=edge['d_inter'][str(2*(nv+1)+1)]+num_eachwindow[str(nv+1)]['ww']
                        nv+=1
                    p_component.append(((w_sp,h_gap),num_eachwindow[str(nv+1)]['ww'],num_eachwindow[str(nv+1)]['hw']))
                    
            else:
                for n in range(nwindow):
                    w_sp=eachfloor[str(i)]['w_bound']+(num_eachwindow['0']['ww']+edge['d_intra'][str(2*(nv+1))])*n
                    p_component.append(((w_sp,h_gap),num_eachwindow['0']['ww'],num_eachwindow['0']['hw'])) 
    return p_component

def plot(W,H,para_set_create): 
    def plot_all(ax,p_component,W,H,xmin,ymin,c):
    
        for (x,y),w,h in p_component:
            rect1=plt.Rectangle((x+xmin,y+ymin),w,h,color=c)
            ax.add_patch(rect1)
        
    p_window2=window_rect_complex(W,H,para_set_create,'window')  
    p_Stairwell_window=window_rect_complex(W,H,para_set_create,'Stairwell_window')  
    p_balcony3=window_rect_complex(W,H,para_set_create,'balcony')
    p_door4=window_rect_complex(W,H,para_set_create,'door')
    
    fig=plt.figure(dpi=120)
    ax = fig.add_subplot(111)
    
    plt.title('SEMANTIC FACADE');
    
    plot_all(ax,p_window2,W,H,0,0,'blue')
    plot_all(ax,p_balcony3,W,H,0,0,'red')
    plot_all(ax,p_door4,W,H,0,0,'green')
    plot_all(ax,p_Stairwell_window,W,H,0,0,'cyan')
    
    ax.set_xlim(0-1,0+W+2)
    ax.set_ylim(0-1,0+H+2)
    
    ax.set_aspect(1)
    plt.show()     
    
def drawrect(imgpath, imgname, img_save, W, H, p_window_o_set, pwindows_set, s):
    imgpath = imgpath+'/'+imgname+'.jpg'
    print (imgpath)
    img = cv2.imread(imgpath)
    # WI = int(W*s)
    # HI = int(H*s)
    for p_window_o in p_window_o_set:
        if p_window_o!=[]:
            print ('drawp_window_o_set',len(p_window_o))
            for (x,y),w,h in p_window_o:
                pt1 = (int(x),H-int(y))
                pt2 = (int((x+w)),H-int((y+h)))
                color = (0,255,255)
                cv2.rectangle(img, pt1, pt2, color,3)
    for pwindows in pwindows_set.values():
        if pwindows!=[]:
            for ((x,y),w,h) in pwindows:
                pt1 = (int(x*s),H-int(y*s))
                pt2 = (int((x+w)*s),H-int((y+h)*s))
                color = (255,255,0) 
                cv2.rectangle(img, pt1, pt2, color,4)
            filename = img_save+'facade_OPT_'+imgname+'2.png'
    else:
        filename = img_save+'opt_facade_'+imgname+'2.png'
    print (filename, img.shape)
    # print ('lastwindow',pt1, pt2) 
    cv2.imwrite(filename, img)
    
def getparaset(x,H,W,s,side):
    p_window=[]
    p_balcony=[]
    p_door=[]
   
    for i in range(len(x[0])):
       if x[1][i]==1:
           p_window.append(((x[0][i][1],H-x[0][i][2]),abs(x[0][i][3]-x[0][i][1]),abs(x[0][i][2]-x[0][i][0])))
       if x[1][i]==2:
           p_balcony.append(((x[0][i][1],H-x[0][i][2]),abs(x[0][i][3]-x[0][i][1]),abs(x[0][i][2]-x[0][i][0])))
       if x[1][i]==3:
           p_door.append(((x[0][i][1],H-x[0][i][2]),abs(x[0][i][3]-x[0][i][1]),abs(x[0][i][2]-x[0][i][0])))
       else:
           continue
    para_set={}
    # p_window.append(((76, 245), 15, 26))
    # p_window.append(((22, 245), 15, 26))
    p_window_set = rightwindow(p_window,H,W,s,'main')
    
    if p_window_set[1]!=[]:
        labelnew = 'Stairwell_window'
        # para_set[labelnew]={}
        # para_set[labelnew]['eachfloor']={}
        eachfloor_Stairwell_window=deducepara(H,p_window_set[1],labelnew,s)
    else:
        eachfloor_Stairwell_window={}
     
    if p_window !=[]:
        eachfloor=deducepara(H,p_window_set[0],'window',s)

    else:
        eachfloor={}
    # if p_balcony !=[]:
    #     eachfloor_balc=deducepara(H,rightwindow(p_balcony,s)[0],'balcony',s)
    # else:
    eachfloor_balc={}
    # if p_door !=[]:
    #     eachfloor_door=deducepara(H,rightwindow(p_door,s)[0],'door',s)
    # else:
    eachfloor_door={}
    
    
    
    para_set_create={'window':{'eachfloor':eachfloor,
                        'connect_floor':
                    {'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'balcony':{'eachfloor':eachfloor_balc,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'door':{'eachfloor':eachfloor_door,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'Stairwell_window':{'eachfloor':eachfloor_Stairwell_window,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}}}     
    
    p_window2=window_rect_complex(W/s,H/s,para_set_create,'window')   
    stairwindow = window_rect_complex(W/s,H/s,para_set_create,'Stairwell_window')   
    
    # plot(W/s,H/s,para_set_create)
    iou = cal_iou([p_window][0], {'window':p_window2, 'Stairwell_window':stairwindow},s)
    print ([p_window][0])
    return p_window, para_set_create, iou

        
#---------------------------------------------

if __name__ == '__main__':
    img_read = r'samresults'
    imgname = 'CIM3_87_main_1'
    rels = joblib.load('CIM2a_40_relsmodel.pkl')
    s = 57.57
    side = 'main'
    # image_path=r'C:\Users\wpc\Desktop\lod223\segment_result\\tex17373_0_splash_20201215T114928.png'
    # para_path=r'C:\Users\wpc\Desktop\lod223\segment_result\\tex17373_0.pkl'
    para_path = img_read + '/' + imgname + '.pkl'
    image_path = img_read + '/' + imgname+'.jpg'
    # get the shape W,H of image
    image = Image.open(image_path)
    W, H = image.size
    
    windows=joblib.load(para_path)
    p_window=[]
    p_balcony=[]
    p_door=[]
    
    for (x,y),w,h in windows:
        # p_window.append([(x,y),w,h])
        p_window.append([(x,H-h-y),w,h])

    para_set={}
    p_window_set = rightwindow(p_window,s,side)
    # p_window_set = [p_window,[]]#rightwindow(p_window,s)
    # W=1000
    # H=1000
    if p_window_set[1]!=[]:
        labelnew = 'Stairwell_window'
        # para_set[labelnew]={}
        # para_set[labelnew]['eachfloor']={}
        eachfloor_Stairwell_window=deducepara(H,p_window_set[1],labelnew,s)
    else:
        eachfloor_Stairwell_window=[]
     
    if p_window !=[]:
        eachfloor=deducepara(H,p_window_set[0],'window',s)
    else:
        eachfloor=[]
    if p_balcony !=[]:
        eachfloor_balc=deducepara(H,rightwindow(p_balcony,s)[0],'balcony',s)
    else:
        eachfloor_balc=[]
    if p_door !=[]:
        eachfloor_door=deducepara(H,rightwindow(p_door,s)[0],'door',s)
    else:
        eachfloor_door=[]
    # if p_door !=[]:
    #     eachfloor_door=deducepara(H,para_set,rightwindow(p_door)[0],'door',s)
    # else:
    #     eachfloor_door=[]
    
    para_set_create={'window':{'eachfloor':eachfloor,
                        'connect_floor':
                    {'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'balcony':{'eachfloor':eachfloor_balc,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'door':{'eachfloor':eachfloor_door,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}},
                     'Stairwell_window':{'eachfloor':eachfloor_Stairwell_window,
                     'connect_floor':{'intra':{'0':0,'2':0,'4':0,'6':0},'d_intra':{'1':0,'3':0,'5':0}}}}
    
    
    
    # para_set_create = getparaset(x,H,W,s)
    p_window2=window_rect_complex(W/s,H/s,para_set_create,'window')   
    stairwindow = window_rect_complex(W/s,H/s,para_set_create,'Stairwell_window')   
    # para_set_create2, iou = getparaset(windows,H,W,s,side)
    plot(W/s,H/s,para_set_create)
    # plot(W/s,H/s,para_set_create2)
    # imgpath = r'rectorig'
    img_save = r'optimage/'
    
    drawrect(img_read, imgname, img_save, W, H, [p_window_set[0]], {'window':p_window2, 'Stairwell_window':stairwindow}, s)
    # drawrect(img_read, imgname, img_save, W, H, [p_window], {'window':p_window2, 'Stairwell_window':stairwindow}, s)
    
    cal_iou([p_window][0], {'window':p_window2, 'Stairwell_window':stairwindow},s)
    






















