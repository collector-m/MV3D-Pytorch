import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def getCalibMatrix(dataPath, frameNum):
    # load calibration data
    # P0, P1, P2, P3, Tr_velo_to_cam, Tr_imu_to_velo
    pathCalib = dataPath+'calib/{:0>6}.txt'.format(frameNum)
    P_left = np.genfromtxt(pathCalib,dtype=None,usecols=range(1,13),skip_header=2,skip_footer=4).reshape(3,4) # 4x4
    rect_3x3 = np.genfromtxt(pathCalib,dtype=None,usecols=range(1,10),skip_header=4,skip_footer=2).reshape(3,3) # 3x3
    velo2cam_3x4 = np.genfromtxt(pathCalib,dtype=None,usecols=range(1,13),skip_header=5,skip_footer=1).reshape(3,4) # 4x4

    rect = np.eye(4)
    velo2cam = np.eye(4)

    rect[:3,:3] =rect_3x3
    velo2cam[:3, :3]  = velo2cam_3x4[:3,:3]
    velo2cam[:3, 3] = velo2cam_3x4[:3, 3]
    
    return {'P_left':P_left,'rect':rect,'velo2cam':velo2cam}

def removePoints(PointCloud, Calib, BoundaryCond):
    # Transform matrix from Calibration
    Tr_p2 = Calib['P_left']; Tr_rect = Calib['rect']; Tr_velo2cam = Calib['velo2cam']

    # Boundary condition
    minX = BoundaryCond['minX'] ; maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY'] ; maxY = BoundaryCond['maxY']
    imgH = BoundaryCond['imgH'] ; imgW = BoundaryCond['imgW'] 
    
    # Remove the point out of range x,y
    #minX = 0;maxX = 70.4;    minY = -40; maxY = 40;
    #imgH = 370; imgW = 1224
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0]<=maxX) & (PointCloud[:, 1] >= minY) & (PointCloud[:, 1]<=maxY))
    PointCloud = PointCloud[mask]

    # Remove the point out of image size
    PointCloud_proj = np.copy(PointCloud)
    PointCloud_proj[:,3] = 1
    PointCloud_proj = np.dot(Tr_velo2cam, PointCloud_proj.T)
    PointCloud_proj = np.dot(Tr_rect, PointCloud_proj)
    PointCloud_proj = np.dot(Tr_p2, PointCloud_proj).T

    PointCloud_z = PointCloud_proj[:,2]
    PointCloud_proj = PointCloud_proj / PointCloud_z[:,None]

    mask = np.where((PointCloud_proj[:,0] >= 1) & (PointCloud_proj[:,0] <= imgW) & (PointCloud_proj[:,1] >=1) & (PointCloud_proj[:,1] <= imgH))    
    PointCloud_proj = PointCloud_proj[mask]
    PointCloud = PointCloud[mask]

    return PointCloud

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    # 704 x 800 x (M+2)
    M = 8; 
    min_z = BoundaryCond['minZ']; max_z = BoundaryCond['maxZ']; gap = (max_z - min_z) / M;
    Height = np.int_(BoundaryCond['maxX'] / Discretization)
    Width = np.int_(BoundaryCond['maxY'] / Discretization) * 2
    BVFeautre = np.zeros((Height,Width, M + 2))

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization))
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width/2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:,2],PointCloud[:,1],PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height,Width,M))        
    for i in range(M):
        lower = min_z + gap * i
        upper = min_z + gap * (i + 1)
        mask_frac = np.where((PointCloud[:,2] >= lower) & (PointCloud[:,2]< upper))
        PointCloud_frac = PointCloud[mask_frac]

        _, indices = np.unique(PointCloud_frac[:,0:2], axis = 0, return_index=True)
        PointCloud_frac = PointCloud_frac[indices]
        
        heightMap[np.int_(PointCloud_frac[:,0]), np.int_(PointCloud_frac[:,1]), i] = PointCloud_frac[:,2]
        """
        plt.imshow(heightMap[:,:,i])
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        """
    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height,Width))
    densityMap = np.zeros((Height,Width))
    
    _, indices, counts = np.unique(PointCloud[:,0:2], axis = 0, return_index=True,return_counts = True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1)/np.log(64))
    
    intensityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = PointCloud_top[:,3]
    densityMap[np.int_(PointCloud_top[:,0]), np.int_(PointCloud_top[:,1])] = normalizedCounts
    """
    plt.imshow(densityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    plt.imshow(intensityMap[:,:])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    """
    output = np.zeros((Height,Width,M+2))
    output[:,:,0:M] = heightMap
    output[:,:,M] = densityMap
    output[:,:,M+1] = intensityMap
    return output
    
def makeFVFeature(PointCloud, FeatureSize):
    height = FeatureSize['height']
    width = FeatureSize['width']

    nPoint = PointCloud.shape[0]
    x = PointCloud[:,0]
    y = PointCloud[:,1]
    z = PointCloud[:,2]
    xy = PointCloud[:,0:2]
    #
    c = np.arctan2(y, x)
    r = np.arctan2(z, np.linalg.norm(xy,axis=1))

    # Normalizing the coordinate of c and r
    # Some issuses occurs when doing ROI pooling
    minC = np.amin(c); maxC = np.amax(c) 
    minR = np.amin(r); maxR = np.amax(r)

    c = np.around((c - minC) * (width - 1) / (maxC - minC)) 
    r = np.around((r - minR) * (height - 1) / (maxR - minR)) 
    
    coordinate = np.int_(np.zeros((nPoint,2)))
    featureValue = np.zeros((nPoint,3))

    coordinate[:,0] = np.int_(r)
    coordinate[:,1] = np.int_(c)
    featureValue[:,0] = z # Height
    featureValue[:,1] = np.linalg.norm(PointCloud[:,0:3],axis=1) # Distance
    featureValue[:,2] = PointCloud[:,3] # Intensity

    #featureValue = featureValue[np.argsort(featureValue[:,1])]
    #featureValue = featureValue[np.argsort(featureValue[:,0])]
    output = np.zeros((height,width,3))

    output[coordinate[:,0],coordinate[:,1],:] = featureValue[:,0:3]
    return output

#    return

# How can I define these in main script
PATH_TO_KITTI = '/home/dongwoo/Project/dataset/KITTI/Object/training/' 
bc={}
bc['minX'] = 0; bc['maxX'] = 70.4; bc['minY'] = -40; bc['maxY'] = 40
bc['minZ'] = - 2.7; bc['maxZ'] = 2.9
bc['imgH'] = 370; bc['imgW'] = 1224
FeatureSize = {}
FeatureSize['height'] = 64
FeatureSize['width'] = 512
# load point cloud data
a = np.fromfile('./000005.bin', dtype=np.float32).reshape(-1, 4)

c = getCalibMatrix(PATH_TO_KITTI, 5)

b = removePoints(a,c,bc)

d = makeFVFeature(b, FeatureSize)
t = time.time()
e = makeBVFeature(b, bc ,0.1)
elapsed = time.time() - t
print ('Dongwoos implementation',elapsed)
