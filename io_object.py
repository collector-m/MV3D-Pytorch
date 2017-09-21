import os
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import time
import math
TRUNC_TRUNCATED = 1

def load_rgb(data_path):
    im_path = os.path.join(data_path,'image_2','*.png')
    im_files = sorted(glob.glob(im_path))

    im_all = []
    
    #tic = time.time()
    for iter_files in im_files:            
        #if len(im_all)<100 :
            im = np.uint8(mpimg.imread(iter_files) * 255)
            im_all.append(im) 
    
    return im_all

def load_velo(data_path):
    velo_path = os.path.join(data_path,'velodyne','*.bin')
    velo_files = sorted(glob.glob(velo_path))

    velo_all = []

    for iter_files in velo_files:
        #if len(velo_all)<100 :
            velo = np.fromfile(iter_files, dtype=np.float32)
            velo_all.append(velo.reshape((-1,4)))

    return velo_all

def load_label(data_path):
    label_path = os.path.join(data_path,'label_2','*.txt')
    label_files = sorted(glob.glob(label_path))

    label_all = []

    for iter_files in label_files:
        #if len(label_all)<100 :
            objects = []
            with open(iter_files, 'r') as file:
                for line in file:
                    object_tmp = line.split()
                    if object_tmp[0] != 'DontCare' and float(object_tmp[1]) != TRUNC_TRUNCATED:
                        h,w,l = list(map(float, object_tmp[8:11]))
                        translation = list(map(float, object_tmp[11:14]))

                        trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
                            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
                            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
                            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
                        yaw = float(object_tmp[14])
                        rotMat = np.array([\
                            [np.cos(yaw), -np.sin(yaw), 0.0], \
                            [np.sin(yaw),  np.cos(yaw), 0.0], \
                            [        0.0,          0.0, 1.0]])
                        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

                        x, y, z = translation
                        yawVisual = ( yaw - np.arctan2(y, x) ) % (2*math.pi)

                        o = type('',(), {})()
                        o.box = cornerPosInVelo.transpose()
                        o.type = object_tmp[0]
                        #o.tracklet_id = n
                        objects.append(o)

                label_all.append(objects)
    return label_all


#Test
"""
basedir = '/home/dongwoo/Project/dataset/KITTI/Object/training'
#date  = '2011_09_26'
#drive = '0005'
#dp = os.path.join(basedir, date,'{}_drive_{}_sync'.format(date,drive))
tic = time.time()
#velo1 = load_velo(basedir)
#rgb = load_rgb(basedir)
label = load_label(basedir)
toc = time.time() - tic
print(toc)

print('Fine')

"""
