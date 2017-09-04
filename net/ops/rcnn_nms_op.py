from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from didi_data.lidar_top      import *
from didi_data.lidar_surround import *


#before nms
def draw_rcnn_berfore_nms(image, probs,  deltas, rois, rois3d, threshold=0.8):

    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>threshold)[0]

    #post processing
    rois   = rois[idx]
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]

    num = len(rois)
    for n in range(num):
        a   = rois[n,1:5]
        cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(rois[:,1:5],deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        draw_box3d_on_top(image,boxes3d)




#after nms : camera image
def draw_rcnn_nms_rgb(rgb, boxes3d, probs, darker=0.7):

    img_rcnn_nms = rgb.copy()*darker
    projections = box3d_to_rgb_projections(boxes3d)
    img_rcnn_nms = draw_rgb_projections(img_rcnn_nms,  projections, color=(255,255,255), thickness=1)

    return img_rcnn_nms

#after nms : lidar top image
def draw_rcnn_after_nms_top(image, boxes3d, probs):
    draw_box3d_on_top(image,boxes3d)



#after nms : lidar surround image
def draw_rcnn_after_nms_surround(image, boxes3d, probs):
    draw_box3d_on_surround(image,boxes3d)





## temporay post-processing ....
## <todo> to be updated
def rcnn_nms( probs,  deltas,  rois3d,  cls=1, threshold = 0.5):

    # do for class-one only
    probs = probs[:,cls]
    idx = np.where(probs>threshold)[0]

    #post processing
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]
    probs  = probs [idx]

    if deltas[0].shape==(4,):
        boxes = box_transform_inv(priors,deltas)
        raise Exception('not implemented !')
        #return probs,boxes

    if deltas[0].shape==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)

        #non-max
        min_dist=0.5
        order = probs.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            #------------------------
            dig  = np.sum(np.sum((boxes3d[i,[0,1,2,3]] -  boxes3d[i,[2,3,0,1]])**2, axis=1)**0.5)/4
            diff = boxes3d[order[1:]] - boxes3d[i]
            dis = np.sum(np.sum(diff**2, axis=2)**0.5, axis=1)/8
            inds = np.where(dis > min_dist*dig)[0]
            order = order[inds + 1]

        probs  =probs[keep]
        boxes3d=boxes3d[keep]

        return probs, boxes3d