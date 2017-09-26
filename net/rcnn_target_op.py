from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *



# gt_boxes    : (x1,y1,  x2,y2  label)  #projected 2d
# gt_boxes_3d : (x1,y1,z1,  x2,y2,z2,  ....    x8,y8,z8,  label)


def rcnn_target(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets


def draw_rcnn_labels(image, rois,  labels, darker=0.7):
    is_print=0

    ## draw +ve/-ve labels ......
    boxes = rois[:,1:5]
    labels = labels.reshape(-1)

    fg_label_inds = np.where(labels != 0)[0]
    bg_label_inds = np.where(labels == 0)[0]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rcnn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    img_label = image.copy()*darker
    if 1:
        for i in bg_label_inds:
            a = boxes[i]
            cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
            cv2.circle(img_label,(a[0], a[1]),2, (32,32,32), -1)

    for i in fg_label_inds:
        a = boxes[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (255,0,255), -1)

    return img_label

def draw_box3d_on_top(image, box3d,color=(255,255,255), thickness=1):

    b   = box3d
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
    cv2.line(image, (u0,v0), (u1,v1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (u1,v1), (u2,v2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (u2,v2), (u3,v3), color, thickness, cv2.LINE_AA)
    cv2.line(image, (u3,v3), (u0,v0), color, thickness, cv2.LINE_AA)


def draw_rcnn_targets(image, rois, labels,  targets, darker=0.7):
    is_print=0

    #draw +ve targets ......
    boxes = rois[:,1:5]

    fg_target_inds = np.where(labels != 0)[0]
    num_pos_target = len(fg_target_inds)
    if is_print: print ('rcnn target : num_pos=%d'  %(num_pos_target))

    img_target = image.copy()*darker
    for n,i in enumerate(fg_target_inds):
        a = boxes[i]
        cv2.rectangle(img_target,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)

        if targets.shape[1:]==(4,):
            t = targets[n]
            b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
            b = b.reshape(4)
            cv2.rectangle(img_target,(b[0], b[1]), (b[2], b[3]), (255,255,255), 1)

        if targets.shape[1:]==(8,3):
            t = targets[n]
            a3d = box_to_box3d(a.reshape(1,4))
            b3d = box3d_transform_inv(a3d, t.reshape(1,8,3))
            b3d = b3d.reshape(8,3)
            draw_box3d_on_top(img_target, b3d)

    return img_target



def draw_rcnn(image, probs,  deltas, rois, threshold=0.75, darker=0.7):

    img_rcnn = image.copy()*darker
    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>0.8)[0]

    #post processing
    priors = rois  [idx,1:5]
    deltas = deltas[idx,cls]

    num = len(priors)
    for n in range(num):
        a   = priors[n]
        cv2.rectangle(img_rcnn,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(priors,deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        priors3d = box_to_box3d(priors)
        boxes3d  = box3d_transform_inv(priors3d, deltas)

        num = len(boxes3d)
        for n in range(num):
            b3d = regularise_box3d(boxes3d[n])
            draw_box3d_on_top(img_rcnn, b3d)


    return img_rcnn




