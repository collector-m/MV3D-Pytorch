import numpy as np

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


# How can I define these in main script
PATH_TO_KITTI = '/home/dongwoo/Project/dataset/KITTI/Object/training/' 
bc={}
bc['minX'] = 0; bc['maxX'] = 70.4; bc['minY'] = -40; bc['maxY'] = 40
bc['imgH'] = 370; bc['imgW'] = 1224

# load point cloud data
a = np.fromfile('./000005.bin', dtype=np.float32).reshape(-1, 4)

c = getCalibMatrix(PATH_TO_KITTI, 5)

b = removePoints(a,c,bc)

print('hello')