import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getCalibMatrix(dataPath, frameNum):
    # load calibration data
    # P0, P1, P2, P3, Tr_velo_to_cam, Tr_imu_to_velo
    pathCalib = 'calib/{:0>6}.txt'.format(frameNum)
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
PATH_TO_KITTI = '~/Desktop/3D/MV3D-Pytorch-master/' 
print(PATH_TO_KITTI)
bc={}
bc['minX'] = 0; bc['maxX'] = 70.4; bc['minY'] = -40; bc['maxY'] = 40
bc['imgH'] = 370; bc['imgW'] = 1224

# load point cloud data
a = np.fromfile('000005.bin', dtype=np.float32).reshape(-1, 4)

c = getCalibMatrix(PATH_TO_KITTI, 5)

b = removePoints(a,c,bc)

#estimation of density
mask1 = np.where((b[:, 0] >= 14) & (b[:, 0]<=15) & (b[:, 1] >= -14) & (b[:, 1]<=15))

nx, ny, nz = (705, 801, 17)
x = np.linspace(0, 70.4, nx)
y = np.linspace(-40, 40, ny)
z = np.linspace(b[:,2].min(), b[:,2].max(), nz)

s = np.zeros((704,800))
density = np.zeros((704,800))
for i in range(0,704):
    for j in range(0,800):
     mask2 = np.where((b[:, 0] >= x[i]) & (b[:, 0]<=x[i+1]) & (b[:, 1] >= y[j]) & (b[:, 1]<=y[j+1]))
     newb = b[mask2]
     s[i,j] = np.size(newb)
     density[i,j] = min(1,(np.log(s[i,j]+1)/np.log(64)))

print(density.max())
plt.imshow(density)
plt.show()


#estimation of intensity
z = 0
intensity = np.zeros((704,800))
for i in range(0,704):
    for j in range(0,800):
     mask3 = np.where((b[:, 0] >= x[i]) & (b[:, 0]<=x[i+1]) & (b[:, 1] >= y[j]) & (b[:, 1]<=y[j+1]))
     newb1 = b[mask3]
     
     if len(newb1[:,2])>0:

      m = newb1[:,2].max()
      n = newb1[:,2].tolist().index(m)
      intensity[i,j] = newb1[n,3]
     else:
         pass
print(intensity.max())
plt.imshow(intensity)
plt.show()

#estimation of height maps
heightmaps = np.zeros((704,800,16))
for k in range(0,16):
    for i in range(0,704):
        for j in range(0,800):
            mask4 = np.where((b[:, 0] >= x[i]) & (b[:, 0]<=x[i+1]) & (b[:, 1] >= y[j]) & (b[:, 1]<=y[j+1]) & (b[:,2]>=z[k]) & (b[:,2]<=z[k+1]))
            newb2 = b[mask4]
            if len(newb2[:,2])>0:
                heightmaps[i,j,k] = newb2[:,2].max()
            else:
             pass
print(heightmaps.max())
