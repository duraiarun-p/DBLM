#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 13:57:04 2021

@author: arun
"""
import sys, getopt
import os
from pathlib import Path
import numpy as np
from random import randint
from scipy.io import savemat
import datetime
# from numba import njit, prange
# import matplotlib
# import matplotlib.pyplot as plt
import time
import cv2
import itk
from pathlib import Path
# from random import randint
import numpy as np
import math
import matlab.engine
import skimage.morphology
from scipy import ndimage
from skimage import measure
from scipy.signal import savgol_filter
import multiprocessing
import concurrent.futures
from traceback import print_exc
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%% Functions as in order

def folderselection(Folderpath):
    allfilelist=[]
    basepath = Path(Folderpath)
    files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_dir())
    mylist=[]
    for item in files_in_basepath:
        mylist.append(item.name)
    Patlen=len(mylist)
    mylist.sort()
    for Pati in range(0,Patlen):
        Patfolderpath=Folderpath+"/"+mylist[Pati]
        # print(Patfolderpath)
        basepath1 = Path(Patfolderpath)
        files_in_basepath1 = (entry for entry in basepath1.iterdir() if entry.is_dir())
        treatlist=[]
        for item in files_in_basepath1:
            treatlist.append(item.name)
        treatlist.sort()
        Tretlen=len(treatlist)
        Plandir=Patfolderpath+"/"+treatlist[0]
        # print(Plandir)
        for Treti in range(1,Tretlen):
            Tretdir=Patfolderpath+"/"+treatlist[Treti]
            # print(Tretdir)
            filelistele=[Plandir,Tretdir,treatlist[Treti]]
            allfilelist.append(filelistele)
    return allfilelist
def selectdirectory(Folderpath):
    basepath = Path(Folderpath)
    #changing is_dir() checks folder
    files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_dir())
    mylist=[]
    for item in files_in_basepath:
        mylist.append(item.name)
    #Random patient folder selection
    PatLen=np.shape(mylist)
    Pati=randint(0,PatLen[0]-1)#Patient folder index not the length
    #Patient Data selected
    Patfolderpath=Folderpath+"/"+mylist[Pati]
    #Planning scan and treatment scan selection
    basepath = Path(Patfolderpath)
    #changing is_dir() checks folder
    files_in_basepath = (entry for entry in basepath.iterdir() if entry.is_dir())
    treatlist=[]
    for item in files_in_basepath:
        treatlist.append(item.name)
    treatlist.sort()
    TreatLen=np.shape(treatlist)
    Treati=randint(1,TreatLen[0]-1)#Patient inter fraction scan folder index not the length
    Plandir=Patfolderpath+"/"+treatlist[0]
    Tretdir=Patfolderpath+"/"+treatlist[Treati]
    print(Plandir)
    print(Tretdir)
    return Plandir,Tretdir
# Image resizing using bicubic interpolation without anti-aliasing in MATLAB the same but with anti-aliasing
def myimresize(IP1a,width,height,Imsiz):
    I1o = np.zeros((width, height,Imsiz[2]))
    for idx in range(Imsiz[2]):
        img=IP1a[:,:,idx]
        I1o[:,:,idx] = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return I1o

def angle_between2(v1, v2):
    with np.errstate(divide='ignore', invalid='ignore'):
        angle=np.nan_to_num(math.degrees(np.arctan(np.vdot(v1,v2)/np.linalg.norm(np.cross(v1,v2)))))
    return angle

def dblmfp1(imgTreatment,imgPlanning,ZLocV,LenX,LenY,p,searchstep,indL):
    costpast=np.ones((LenX,LenY))*costV
    mV1m=np.ones((LenX,LenY))*-p
    mV1n=np.ones((LenX,LenY))*-p
    mV1z=np.ones((LenX,LenY))*0
    mV1Z=np.ones((LenX,LenY))*ZLocV[0]
    mn=mbSize*mbSize
    mbCountL=LenX*LenY
    # The following loop is the right way of calculating distance and cost
    for zi in range(0,indL):   
        for m1 in np.arange(-p,p+searchstep,searchstep):
            for n1 in np.arange(-p,p+searchstep,searchstep):
                
                absdiff=np.zeros((m,n))
                absdiff[max(-m1,0):min(m,m-m1),max(-n1,0):min(n,n-n1)] = \
                          abs(imgTreatment[max(-m1,0):min(m,m-m1),max(-n1,0):min(n,n-n1)] - \
                          imgPlanning[max(m1,0):min(m,m+m1),max(n1,0):min(n,n+n1),zi])
    
                err=absdiff.reshape(-1,mbSize).sum(axis=1).reshape(LenY,mbSize,-1).sum(axis=1)
                
                dist1=(m1**2+n1**2)**0.5+(abs(ZLocV[zi])*10)
                cost=(err/(mn))+lambdadist*dist1
                dispm=abs(m1//mbSize)
                dispn=abs(n1//mbSize)                
                
                if m1==0 and n1==0:
                    pass
                elif m1<0 and n1<0:
                      cost[0:0+dispm,0:]=costV
                      cost[0:,0:0+dispn]=costV
                elif m1<0 and n1==0:
                      cost[0:0+dispm,0:]=costV
                elif m1<0 and n1>0:
                      cost[0:0+dispm,0:]=costV
                      cost[0:,-dispn:]=costV
                elif m1==0 and n1<0:
                      cost[0:,0:0+dispn]=costV
                elif m1==0 and n1>0:
                      cost[0:,-dispn:]=costV
                elif m1>0 and n1<0:
                      cost[-dispm:,0:]=costV
                      cost[0:,0:0+dispn]=costV
                elif m1>0 and n1==0:
                      cost[-dispm:,0:]=costV
                elif m1>0 and n1>0:
                      cost[-dispm:,0:]=costV
                      cost[0:,-dispn:]=costV                
                # cost=(err/(25))
                costind=cost<costpast
                #check minimum here and change the existing vectors with new value
                mV1m[costind]=m1
                mV1n[costind]=n1
                mV1z[costind]=zi
                mV1Z[costind]=ZLocV[zi]
                # print('1st block m1=%s n1=%s zi= %s dist1=%s costpast=%s cost=%s mV1m=%s'%(m1,n1,zi,dist1,costpast[0],cost[0],mV1m[0]))
                costpast[costind]=cost[costind]
    mV1mr=mV1m.reshape((mbCountL,1))
    mV1nr=mV1n.reshape((mbCountL,1))
    mV1Zr=mV1Z.reshape((mbCountL,1))
    mV1zr=mV1z.reshape((mbCountL,1))
    mV11=np.hstack((mV1mr,mV1nr,mV1Zr))
    P1=np.array([int(m1),int(n1),int(ZLocV[zi])],dtype=int).reshape(1,3)#Reference vector for angle calculation
    omg=[]
    for mbc in range(mbCountL):
        P2=mV11[mbc,:]
        omge=angle_between2(P1, P2)
        omg.append(omge) 
    mV1=np.hstack((mV1mr,mV1nr,mV1zr))
    omg2a=np.asarray(omg).reshape(LenX,LenY)
    return mV1,omg2a
def omgabsdiff2(omeg,LenX,LenY):
    abssum=np.zeros((LenX,LenY))
    for m1x in range(-1,2):
         for n1x in range(-1,2):
             if m1x==0 and n1x==0:
                 continue
             abssum[max(m1x,0):min(LenX,LenX+m1x),max(n1x,0):min(LenY,LenY+n1x)]  \
                 = abssum[max(m1x,0):min(LenX,LenX+m1x),max(n1x,0):min(LenY,LenY+n1x)] \
                     + abs(omeg[max(m1x,0):min(LenX,LenX+m1x),max(n1x,0):min(LenY,LenY+n1x)] \
                           - omeg[max(-m1x,0):min(LenX,LenX-m1x),max(-n1x,0):min(LenY,LenY-n1x)])
    abssum=abssum/9
    return abssum

def dblmsp1(imgTreatment,imgPlanning,ZLocV,LenX,LenY,p,searchstep,indL,omeg):
    costpast2=np.ones((LenX,LenY))*costV
    mV2m=np.ones((LenX,LenY))*-p
    mV2n=np.ones((LenX,LenY))*-p
    mV2z=np.ones((LenX,LenY))*0
    mV2Z=np.ones((LenX,LenY))*ZLocV[0]
    mn=mbSize*mbSize
    mbCountL=LenX*LenY
    # The following loop is the right way of calculating distance and cost
    for zi in range(0,indL):   
        for m1 in np.arange(-p,p+searchstep,searchstep):
            for n1 in np.arange(-p,p+searchstep,searchstep):
                
                absdiff=np.zeros((m,n))
                absdiff[max(-m1,0):min(m,m-m1),max(-n1,0):min(n,n-n1)] = \
                          abs(imgTreatment[max(-m1,0):min(m,m-m1),max(-n1,0):min(n,n-n1)] - \
                          imgPlanning[max(m1,0):min(m,m+m1),max(n1,0):min(n,n+n1),zi])
    
                err=absdiff.reshape(-1,mbSize).sum(axis=1).reshape(LenY,mbSize,-1).sum(axis=1)
                
                dist1=(m1**2+n1**2)**0.5+(abs(ZLocV[zi])*10)
                cost2=(err/(mn))+lambdadist*dist1
                dispOm=m1//mbSize
                dispOn=n1//mbSize
                
                cost2[max(-dispOm,0):min(LenX,LenX-dispOm),max(-dispOn,0):min(LenY,LenY-dispOn)] = \
                          cost2[max(-dispOm,0):min(LenX,LenX-dispOm),max(-dispOn,0):min(LenY,LenY-dispOn)] + \
                          (lambdaomeg*omeg[max(dispOm,0):min(LenX,LenX+dispOm),max(dispOn,0):min(LenY,LenY+dispOn)])
                dispm=abs(dispOm)
                dispn=abs(dispOn)
                if m1==0 and n1==0:
                    pass
                elif m1<0 and n1<0:
                      cost2[0:0+dispm,0:]=costV
                      cost2[0:,0:0+dispn]=costV
                elif m1<0 and n1==0:
                      cost2[0:0+dispm,0:]=costV
                elif m1<0 and n1>0:
                      cost2[0:0+dispm,0:]=costV
                      cost2[0:,-dispn:]=costV
                elif m1==0 and n1<0:
                      cost2[0:,0:0+dispn]=costV
                elif m1==0 and n1>0:
                      cost2[0:,-dispn:]=costV
                elif m1>0 and n1<0:
                      cost2[-dispm:,0:]=costV
                      cost2[0:,0:0+dispn]=costV
                elif m1>0 and n1==0:
                      cost2[-dispm:,0:]=costV
                elif m1>0 and n1>0:
                      cost2[-dispm:,0:]=costV
                      cost2[0:,-dispn:]=costV  
                          
                costind2=cost2<costpast2
                #check minimum here and change the existing vectors with new value
                mV2m[costind2]=m1
                mV2n[costind2]=n1
                mV2z[costind2]=zi
                mV2Z[costind2]=ZLocV[zi]
                # print('1st block m1=%s n1=%s zi= %s dist1=%s costpast=%s cost=%s mV1m=%s'%(m1,n1,zi,dist1,costpast[0],cost[0],mV1m[0]))
                costpast2[costind2]=cost2[costind2]
    mV2mr=mV2m.reshape((mbCountL,1))
    mV2nr=mV2n.reshape((mbCountL,1))
    mV2Zr=mV2Z.reshape((mbCountL,1)) 
    mV2zr=mV2z.reshape((mbCountL,1)) 
    mV2=np.hstack((mV2mr,mV2nr,mV2zr,mV2Zr))
    return mV2
def motioncomp(m,n,mbSize,mV2,masPlanning):
    imageComp=np.zeros((m,n))
    imageComp=imageComp.astype(int)
    mbCount=0
    for mi in  range(0,m,mbSize):
        for mj in range(0,n,mbSize):
            dy=int(mV2[mbCount,0])
            dx=int(mV2[mbCount,1])
            refBlkVerm = mi + dy
            refBlkHorm = mj + dx
            # refBlkVerm = max(mi + dy,0)
            # refBlkHorm = max(mj + dx,0)
            refBlkDeptm = int(mV2[mbCount,2])
            imageComp[mi:mi+mbSize,mj:mj+mbSize] = masPlanning[refBlkVerm:refBlkVerm+mbSize, refBlkHorm:refBlkHorm+mbSize,refBlkDeptm]
            mbCount=mbCount+1
    return imageComp
def dustthemask(MTE):
    MTE=np.uint8(MTE)
    is_all_zero = np.all((MTE == 0))
    kernel=skimage.morphology.diamond(8)
    Mn=cv2.morphologyEx(MTE,cv2.MORPH_OPEN,kernel)
    Mn1=ndimage.binary_fill_holes(Mn)
    Mc=measure.find_contours(Mn1,0.8)
    if is_all_zero==True or len(Mc)==0:
        Mn1=np.zeros(MTE.shape)
    else:
        Mc1=Mc[0]
        windowWidth = 25 # windowlength must be odd always
        polynomialOrder = 2
        polx = savgol_filter(Mc1[:,0], windowWidth, polynomialOrder, mode='nearest')
        poly = savgol_filter(Mc1[:,1], windowWidth, polynomialOrder, mode='nearest')
        polygon=np.column_stack((polx,poly))
        Mn1 = skimage.draw.polygon2mask(MTE.shape, polygon)
    return Mn1
def dustthemask1(MTE):
    MTE=np.uint8(MTE)
    is_all_zero = np.all((MTE == 0))
    kernel=skimage.morphology.diamond(4)
    Mn=cv2.morphologyEx(MTE,cv2.MORPH_CLOSE,kernel)
    Mn=cv2.morphologyEx(Mn,cv2.MORPH_OPEN,kernel)
    Mn1=ndimage.binary_fill_holes(Mn)
    Mc=measure.find_contours(Mn1,0.8)
    if is_all_zero==True or len(Mc)==0:
        Mn1=np.zeros(MTE.shape)
    else:
        Mc1=Mc[0]
        windowWidth = 35 # windowlength must be odd always
        polynomialOrder = 2
        polx = savgol_filter(Mc1[:,0], windowWidth, polynomialOrder, mode='nearest')
        poly = savgol_filter(Mc1[:,1], windowWidth, polynomialOrder, mode='nearest')
        polygon=np.column_stack((polx,poly))
        Mn1 = skimage.draw.polygon2mask(MTE.shape, polygon)
    return Mn1
def dblmparaprocess(mylist):
    # mylist=inputlist[ind]#data index slice number
    imgTreatment=mylist[0]
    imgPlanning=mylist[1]
    masPlanning=mylist[2]
    ZLocV=mylist[3]
    indL=mylist[4]
    indz=mylist[5]
    lambdadist=mylist[6]
    lambdaomeg=mylist[7]
    LenX=mylist[8]
    LenY=mylist[9]
    m=mylist[10]
    n=mylist[11]
    searchstep=mylist[12]
    p=mylist[13]
    mbSize=mylist[14]
    costV=mylist[15]
    imageComp=mylist[16]
    mV11,omg2a=dblmfp1(imgTreatment,imgPlanning,ZLocV,LenX,LenY,p,searchstep,indL)
    omeg=omgabsdiff2(omg2a,LenX,LenY)
    mV2=dblmsp1(imgTreatment,imgPlanning,ZLocV,LenX,LenY,p,searchstep,indL,omeg)
    MTE1=motioncomp(m,n,mbSize,mV2,masPlanning)
    ITE=motioncomp(m,n,mbSize,mV2,imgPlanning)
    # MTE=MTE1
    MTE=dustthemask(MTE1)
    return MTE, ITE
#Dice OK 
def mydice(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = np.logical_and(mask_gt, mask_pred).sum()
  return 2*volume_intersect / volume_sum 
#This function has been used in the performance evaluation so for now we can use for comparison
def mynmse(a,b):
    aprems=np.linalg.norm(a-b)
    amse=(aprems*aprems)/np.size(a)
    anmse=amse/math.sqrt(math.sqrt(np.sum(a))*math.sqrt(np.sum(b)))
    return anmse  
# This is the right way of calculating normalised mean square error
def mynmse1(a,b):
    aprems=np.linalg.norm(a-b)
    amse=np.sum(aprems*aprems)/np.size(a)
    Ea=np.sum((a*a))/np.size(a)
    Eb=np.sum((b*b))/np.size(b)
    anmse=amse/(math.sqrt(Ea)*math.sqrt(Eb))
    return anmse
def dblmoverall(Z1a,IP1a,IT1a,MP2a,MT2a,lambdadist,lambdaomeg,indSize,mbSize,p,m,n,costV):
    st0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    start_time=time.time()
    IP1a = np.copy(IP1a._data).reshape(IP1a.size, order='F')
    IT1a = np.copy(IT1a._data).reshape(IT1a.size, order='F')
    MP2a = np.copy(MP2a._data).reshape(MP2a.size, order='F')
    MT2a = np.copy(MT2a._data).reshape(MT2a.size, order='F')
    Z1a = np.copy(Z1a._data).reshape(Z1a.size, order='F')
    slicestep=1
    searchstep=1
    LenX=m//mbSize
    LenY=n//mbSize
    mbCountL=LenX*LenX
    Imsiz=IP1a.shape#
    IP1=myimresize(IP1a,n,m,Imsiz)
    IT1=myimresize(IT1a,n,m,Imsiz)
    MP2=myimresize(MP2a,n,m,Imsiz)
    MT2=myimresize(MT2a,n,m,Imsiz)
    num_cores = multiprocessing.cpu_count()
    print('Input list process started')
    mylist=[]
    inputlist=[]
    for ind in range(0,Imsiz[2]):
        indzT=np.arange(max(0-ind,-math.floor(indSize/2)),min(Imsiz[2]-ind-slicestep,math.floor(indSize/2))+slicestep,slicestep)
        indL=indzT.size
        indz=ind+indzT
        Zloc=Z1a[indz]
        ZLocV=Zloc-Zloc[indzT==0]
        imgTreatment=IT1[:,:,ind]
        imgPlanning=IP1[:,:,indz]
        masPlanning=MP2[:,:,indz]
        imageComp=np.zeros((m,n))
        imageComp=imageComp.astype(int)
        mylist=[imgTreatment,imgPlanning,masPlanning,ZLocV,indL,indz,lambdadist,lambdaomeg,LenX,LenY,m,n,searchstep,p,mbSize,costV,imageComp]
        inputlist.append(mylist) 
    reslist=[]
    resparlist=[]
    MTE = np.zeros((m, n, Imsiz[2]))
    ITE = np.zeros((m, n, Imsiz[2]))
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print('Parallel process started at')
    print(st)
    start_timeN=time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for mylist in range(0,Imsiz[2]):
            res=executor.submit(dblmparaprocess,inputlist[mylist])
            reslist.append(res)
        for resi in reslist:
            try:
                result = resi.result()
                resparlist.append(result)
            except:
                resparlist.append(None)
                print_exc()
    runtimeN=(time.time()-start_timeN)
    print('Parfor Time =%s sec'%(runtimeN))
    MTE = np.zeros((m, n, Imsiz[2]))
    ITE = np.zeros((m, n, Imsiz[2]))
    for ind in range(0,Imsiz[2]):
        RES=resparlist[ind]
        MTE[:,:,ind]=RES[0]
        ITE[:,:,ind]=RES[1]

    # for ind in range(0,Imsiz[2]):
    #     print(ind)
    #     MTE1,ITE1=dblmparaprocess(inputlist[ind])
    #     MTE[:,:,ind]=MTE1
    #     ITE[:,:,ind]=ITE1
    
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print('Parallel Process ended at')
    print(st) 
    print('DBLM Process started at')
    print(st0)
    runtimeN=(time.time()-start_time)
    print('DBLM Total Time =%s sec'%(runtimeN))
    Dice=mydice(MTE,MT2)
    Nmse=mynmse(ITE, IT1)
    Nmse1=mynmse1(ITE, IT1)
    print(Plandir)
    print(Tretdir)
    print('Dice score = %s  MatNMSE=%s NMSE=%s '%(Dice,Nmse,Nmse1))
    # print('Dice score = %s '%(Dice))
    return ITE,MTE,IT1,MT2,IP1,MP2,Dice,Nmse,Nmse1
#%%


inputfile = ''
outputfile = ''
try:
  argv=sys.argv[1:]
  opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print(' Check syntax: test.py -i <inputfile> -o <outputfile>')
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit()
  elif opt in ("-i", "--ifile"):
      inputfile = arg
  elif opt in ("-o", "--ofile"):
      outputfile = arg
print('Input file is :', inputfile)
print('Output file is :', outputfile)


#%%
lambdadist=0.0905
lambdaomeg=0.0005
indSize=5
mbSize=5
p=10
costV=np.finfo(np.float64).max
m=500
n=500
Folderpath='/home/arun/Documents/MATLAB/ImageDB/Bladder'
folderlis=folderselection(Folderpath)
DataLen=len(folderlis)
OP=[]
DP=[]
#%%
eng = matlab.engine.start_matlab()
#eng.addpath('/home/s1785969/RDS/MATLAB/DAtoolbox')
#eng.addpath('/home/s1785969/RDS/MATLAB/BlockMatchingAlgoMPEG')
eng.addpath('/home/arun/Documents/MATLAB/')
#eng.addpath('/home/s1785969/RDS/MATLAB/DA_Image_REG')
eng.addpath('/home/arun/Documents/MATLAB/DicomGUI')
eng.addpath('/home/arun/Documents/MATLAB/DicomGUI/Scripting')
eng.addpath('/home/arun/Documents/MATLAB/MAWS_Precision')
eng.addpath('/home/arun/Documents/PyWSPrecision/dblm')
path0="/home/arun/Documents/MATLAB/ImageOutputs/PyOutputs/run1/"
matop="N5p10"
path=path0+matop
os.chdir(path)
datai=5

# for datai in range(0,DataLen):
# Plandir=folderlis[datai][0]
# Tretdir=folderlis[datai][1]
# TretData=folderlis[datai][2]

Plandir=inputfile
Tretdir=outputfile

Z1a,IP1a,IT1a,MP2a,MT2a=eng.blarloadpycall(Plandir,Tretdir,nargout=5)
ITE,MTE,IT1,MT2,IP1,MP2,Dice,Nmse,Nmse1=dblmoverall(Z1a,IP1a,IT1a,MP2a,MT2a,lambdadist,lambdaomeg,indSize,mbSize,p,m,n,costV)
# OPele=[TretData,Dice]
# OP.append(OPele)
# DP.append(Dice)
# OPpath=path+"/"+TretData
# try:
#     os.mkdir(OPpath)
#     os.chdir(OPpath)
# except OSError:
#     os.chdir(OPpath)
# opmatfile=TretData+"OP"+".mat"
# imgdic={"IP1":IP1,"MP2":MP2,"MTE":MTE,"ITE":ITE,"IT1":IT1,"MT2":MT2,"Dice":Dice,"MatNMSE":Nmse,"NMSE":Nmse1}
# savemat(opmatfile, imgdic)
# MTEimg1=TretData+"MTE"+".nrrd"
# ITEimg1=TretData+"ITE"+".nrrd"
# MT2img1=TretData+"MT2"+".nrrd"
# IT1img1=TretData+"IT1"+".nrrd"
# MTEimg=itk.GetImageFromArray(MTE)
# itk.imwrite(MTEimg, MTEimg1)
# ITEimg=itk.GetImageFromArray(ITE)
# itk.imwrite(ITEimg, ITEimg1)
# MT2img=itk.GetImageFromArray(MT2)
# itk.imwrite(MT2img, MT2img1)
# IT1img=itk.GetImageFromArray(IT1)
# itk.imwrite(IT1img, IT1img1)
# os.chdir(path)
eng.quit()
# DP=np.asarray(DP)
# dicematfile="Dice"+matop+".mat"
# savemat(dicematfile,{"Diceall":DP})
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
print('Script Total Time MATLAB Engine included =%s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)
#%%
