from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *
from net.processing.boxes3d import *
from net.utility.draw import *
from net.processing.boxes3d import box3d_to_top_box
import numpy as np
from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *
import time
from net.utility.draw import *
import io_object
#from mayavi import mlab
#mlab.init_notebook()
#mlab.test_contour3d()


# run functions --------------------------------------------------------------------------
MATRIX_Mt = np.array([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
                      [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
                      [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
                      [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])
MATRIX_Kt = np.array([[ 721.5377,    0.    ,    0.    ],
                      [   0.    ,  721.5377,    0.    ],
                      [ 609.5593, 172.854 , 1. ]])

def draw_projected_box3d( image, qs, color=(255,255,255), thickness=2):

     for k in range(0,4):
        #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i,j=k,(k+1)%4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        i,j=k+4,(k+1)%4 + 4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        i,j=k,k+4
        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)


def make_projected_box3d(box3d, Mt=None, Kt=None):

    if Mt is None: Mt = MATRIX_Mt
    if Kt is None: Kt = MATRIX_Kt

    Ps = np.hstack(( box3d, np.ones((8,1))) )
    Qs = np.matmul(Ps,Mt)
    Qs = Qs[:,0:3]
    qs = np.matmul(Qs,Kt)
    zs = qs[:,2].reshape(8,1)
    qs = (qs/zs).astype(np.int32)

    return qs

def obj_to_gt(objs):

    num         = len(objs)
    gt_boxes   = np.zeros((num,4),  dtype=np.float32)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels   = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)

        gt_labels[n]=label
        gt_boxes[n]=np.array([umin,vmin,umax,vmax])
        gt_boxes3d[n]=b

    return gt_labels, gt_boxes, gt_boxes3d

def box_to_box3d(boxes):

    num=len(boxes)
    boxes3d = np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        x1,y1,x2,y2 = boxes[n]

        points = [ (x1,y1), (x1,y2), (x2,y2), (x2,y1) ]
        for k in range(4):
            xx,yy = points[k]
            x,y  = top_to_lidar_coords(xx,yy)
            boxes3d[n,k,  :] = x,y,0.4
            boxes3d[n,4+k,:] = x,y,-2

    return boxes3d



## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels


def lidar_to_front(PointCloud, FeatureSize):
    height = FeatureSize['height']
    width = FeatureSize['width']


    nPoint = PointCloud.shape[0]
    x = PointCloud[:, 0]
    y = PointCloud[:, 1]
    z = PointCloud[:, 2]
    xy = PointCloud[:, 0:2]
    #
    c = np.arctan2(y, x)
    r = np.arctan2(z, np.linalg.norm(xy, axis=1))

    # Normalizing the coordinate of c and r
    # Some issuses occurs when doing ROI pooling
    minC = np.amin(c);
    maxC = np.amax(c)
    minR = np.amin(r);
    maxR = np.amax(r)

    c = np.around((c - minC) * (width - 1) / (maxC - minC))
    r = np.around((r - minR) * (height - 1) / (maxR - minR))

    coordinate = np.int_(np.zeros((nPoint, 2)))
    featureValue = np.zeros((nPoint, 3))

    coordinate[:, 0] = np.int_(r)
    coordinate[:, 1] = np.int_(c)
    featureValue[:, 0] = z  # Height
    featureValue[:, 1] = np.linalg.norm(PointCloud[:, 0:3], axis=1)  # Distance
    featureValue[:, 2] = PointCloud[:, 3]  # Intensity

    # featureValue = featureValue[np.argsort(featureValue[:,1])]
    # featureValue = featureValue[np.argsort(featureValue[:,0])]
    output = np.zeros((height, width, 3))

    output[coordinate[:, 0], coordinate[:, 1], :] = featureValue[:, 0:3]


    front_image = np.sum(output,axis=2)
    front_image = front_image-np.min(front_image)
    front_image = (front_image/np.max(front_image)*255)
    front_image = np.dstack((front_image, front_image, front_image)).astype(np.uint8)


    return output, front_image




## lidar to top ##
def lidar_to_top(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count

                pass
            pass
        pass
    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 1: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


    return top, top_image


## drawing ####

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())



def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    
    basedir = '/home/dongwoo/Project/dataset/KITTI/Object/training'
    outdir = '/home/dongwoo/Project/MV3D/MV3D-mohsen/data'

    """
    date  = '2011_09_26'
    drives = ['0001', '0002', '0029', '0005', '0009', '0011', '0013', '0014', '0017', '0018',
                                   '0048', '0051', '0056', '0057', '0059', '0060', '0084', '0091', '0093']

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive) #, range(0, 50, 5))

    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples
    dataset.load_timestamps()    # Timestamps are parsed into datetime objects
    dataset.load_oxts()          # OXTS packets are loaded as named tuples
    dataset.load_gray()         # Left/right images are accessible as named tuples
    dataset.load_rgb()          # Left/right images are accessible as named tuples
    dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = basedir +'/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date,date,drive)
    """
    dataset_rgb = io_object.load_rgb(basedir)
    dataset_velo = io_object.load_velo(basedir)
    objects = io_object.load_label(basedir)
    num_frames=len(dataset_velo)  #154
    #objects = read_objects(tracklet_file, num_frames)

    ############# convert   ###########################  ************************************
    #os.makedirs('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive)



    if 1:  ## rgb images --------------------
        os.makedirs(outdir+'/rgb/')

        for n in range(num_frames):
            print(n)
            rgb = dataset_rgb[n]
            #rgb =(rgb*255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(outdir+'/rgb/rgb_%05d.png'%n,rgb)


    if 1:  ## front images --------------------
        os.makedirs(outdir+'/front')
        os.makedirs(outdir+'/front_image')


        for n in range(num_frames):
            print(n)
            FeatureSize = {}
            FeatureSize['height'] = 64
            FeatureSize['width'] = 512
            lidar = dataset_velo[n]
            front, front_image = lidar_to_front(lidar,FeatureSize)
            cv2.imwrite(outdir+'/front_image/front_image_%05d.png'%n,front_image)
            np.save(outdir+'/front/front_%05d.npy'%n,front)


    if 1:  ## top images --------------------
        #os.makedirs('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive+'/lidar')
        #os.makedirs('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive+'/top')
        #os.makedirs('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive+'/top_image')

        for n in range(num_frames):
            print(n)
            lidar = dataset_velo[n]
            top, top_image = lidar_to_top(lidar)

            #np.save('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive+'/lidar/lidar_%05d.npy'%n,lidar)
            #np.save('/home/mohsen/Desktop/didi-udacity-2017-master/data/seg/'+drive+'/top/top_%05d.npy'%n,top)
            cv2.imwrite(outdir+'/top_image/top_image_%05d.png'%n,top_image)

        #exit(0)



    if 1:  ## boxes3d  --------------------
        os.makedirs(outdir+'/gt_boxes3d')
        os.makedirs(outdir+'/gt_labels')
        for n in range(num_frames):
            print(n)
            objs = objects[n]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)

            np.save(outdir+'/gt_boxes3d/gt_boxes3d_%05d.npy'%n,gt_boxes3d)
            np.save(outdir+'/gt_labels/gt_labels_%05d.npy'%n,gt_labels)

        #exit(0)


    ############# analysis ########################### *************************************
    if 1: ## make mean
        mean_image = np.zeros((400,400),dtype=np.float32)
        num_frames=20
        for n in range(num_frames):
            print(n)
            top_image = cv2.imread(outdir+'/top_image/top_image_%05d.png'%n,0)
            mean_image += top_image.astype(np.float32)

        mean_image = mean_image/num_frames
        cv2.imwrite(outdir+'/top_image/top_mean_image.png',mean_image)


    if 1: ## gt_3dboxes distribution ... location and box, height
        depths =[]
        aspects=[]
        scales =[]
        mean_image = cv2.imread(outdir+'/top_image/top_mean_image.png',0)

        for n in range(num_frames):
            print(n)
            gt_boxes3d = np.load(outdir+'/gt_boxes3d/gt_boxes3d_%05d.npy'%n)

            top_boxes = box3d_to_top_box(gt_boxes3d)
            draw_box3d_on_top(mean_image, gt_boxes3d,color=(255,255,255), thickness=1, darken=1)


            for i in range(len(top_boxes)):
                x1,y1,x2,y2 = top_boxes[i]
                w = math.fabs(x2-x1)
                h = math.fabs(y2-y1)
                area = w*h
                s = area**0.5
                scales.append(s)

                a = w/h
                aspects.append(a)

                box3d = gt_boxes3d[i]
                d = np.sum(box3d[0:4,2])/4 -  np.sum(box3d[4:8,2])/4
                depths.append(d)

        depths  = np.array(depths)
        aspects = np.array(aspects)
        scales  = np.array(scales)

        np.savetxt(outdir+'/depths.txt',depths)
        np.savetxt(outdir+'/aspects.txt',aspects)
        np.savetxt(outdir+'/scales.txt',scales)
        cv2.imwrite(outdir+'/top_image/top_rois.png',mean_image)








    #----------------------------------------------------------
    #----------------------------------------------------------
    #exit(0)





    #----------------------------------------------------------
    lidar = dataset_velo[0]

    objs = objects[0]
    gt_labels, gt_boxes, gt_boxes3d = obj_to_gt(objs)

#    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
#    draw_lidar(lidar, fig=fig)
    #draw_gt_boxes3d(gt_boxes3d, fig=fig)
    #mlab.show(1)

    print ('** calling lidar_to_tops() **')
    if 1:
        top, top_image = lidar_to_top(lidar)
        rgb = dataset_rgb[0]
    else:
        top = np.load('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/top.npy')
        top_image = cv2.imread('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/top_image.png')
        rgb = np.load('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/rgb.npy')

    #rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # -----------



    #check
    num = len(gt_boxes)
    for n in range(num):
        x1,y1,x2,y2 = gt_boxes[n]
        cv2.rectangle(top_image,(x1,y1), (x2,y2), (0,255,255), 1)


    ## check
    boxes3d0 = box_to_box3d(gt_boxes)

    draw_gt_boxes3d(boxes3d0,  color=(1,1,0), line_width=1, fig=fig)
    #mlab.show(1)

    for n in range(num):
        qs = make_projected_box3d(gt_boxes3d[n])
        draw_projected_box3d(rgb,qs)

    #imshow('rgb',rgb)
    #cv2.waitKey(0)




    #save
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy',rgb)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/lidar.npy',lidar)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/top.npy',top)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.png',top_image)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.maked.png',top_image)

 #  np.save('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/gt_labels.npy',gt_labels)
  #  np.save('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/gt_boxes.npy',gt_boxes)
 #   np.save('/home/mohsen/Desktop/didi-udacity-2017-master/data/one_frame/gt_boxes3d.npy',gt_boxes3d)

    #imshow('top_image',top_image)
    #cv2.waitKey(0)

    #pause
    #exit(0)


 


