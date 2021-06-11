import numpy as np
from VNSC import VoxelNystromSC
import mahotas
import gdal
import gc
import os

####################################### CHM local maximum #######################
def localmaxima(img,d,ws): 
    threshed=(img>ws)
    img *=threshed 
    bc=np.ones((int(ws/d),int(ws/d))) 
    maxima=mahotas.morph.regmax(img,Bc=bc) 
    spots,n_spots=mahotas.label(maxima)
    return n_spots
    
############################################ main ##############################################
path=os.getcwd()
isExists=os.path.exists(path+'\\results')
if not isExists:
    os.mkdir('results')

for root,dirs,files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[1]=='.tif':  
            CHM_name=file
            break  
    for file in files:
        if os.path.splitext(file)[1]=='.txt':
            print('Start:',file)
            X=np.loadtxt(file)
            Z=X[:,2]
            id0=[]
            for ii in range(0,len(Z)):
                if Z[ii]>2.5:
                   id0.extend([ii])
            X=X[id0,:]
            ######################################################
            
            #read and process CHM
            img=gdal.Open(CHM_name)
            im_width=img.RasterXSize 
            im_height=img.RasterYSize 
            im_geotrans=img.GetGeoTransform() 
            x0=im_geotrans[0]  
            y1=im_geotrans[3] 
            d=im_geotrans[1]  
            xi=int((X[np.lexsort([X[:,0]])[0],0]-x0)/d)  
            xj=int((X[np.lexsort([-X[:,0]])[0],0]-x0)/d)  
            dx=xj-xi                                     
            yi=int((y1-X[np.lexsort([X[:,1]])[0],1])/d)
            yj=int((y1-X[np.lexsort([-X[:,1]])[0],1])/d) 
            dy=yj-yi
            xl=max(xi,0)  
            xr=max(min(xj,im_width),0)
            yt=max(yj,0)
            yb=max(min(yi,im_height),0)
            im_data=img.ReadAsArray(xl,yt,xr-xl,yb-yt) 
            nmax=localmaxima(im_data,d,3) 
            del img,im_data
            gc.collect()  
            
            gap=nmax*1.5
            XX=X[:,:3]       
            VoxelNystromSC(XX,file.split('.')[0],int(gap),path)  

