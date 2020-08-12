import numpy as np
import utils

def get_VDG(box1, box2):
    '''
    Visual Dependency Grammar
    input: 2 bbox
    output: spatial relation between 2 boxes
    size of image 785 x 1024 (width = 1024, height = 785)
    '''
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    box1_overlay = utils.percent_overlap(box1, box2)
    box2_overlay = utils.percent_overlap(box2, box1)
    iou = utils.get_iou(box1, box2)
    
    if box2_overlay > 0.9:
        # box2 is in box1 --> box1 - '13':'covering' - box2
        predicate = 'covering'
        encode = '13'
        return (predicate, encode)

    if box1_overlay > 0.5:
        # box1 - '31':'on' - box2
        predicate = 'on'
        encode = '31'
        return (predicate, encode)
    
    box_np1 = np.asarray(box1)
    box_np2 = np.asarray(box2)
    centroid1 = np.asarray([(box1[0] + box1[2])/2, (box1[1] + box1[3])/2])
    centroid2 = np.asarray([(box2[0] + box2[2])/2, (box2[1] + box2[3])/2])
    vec_centroid = centroid2 - centroid1
    vec_anchor = np.asarray([1, 0])

    unit_vector_1 = vec_centroid / np.linalg.norm(vec_centroid)
    unit_vector_2 = vec_anchor / np.linalg.norm(vec_anchor)
    dot_product = np.dot(unit_vector_1, unit_vector_2) # cos(alpha)
    thres_cos_1 = np.cos(45*np.pi/180)
    thres_cos_2 = np.cos(135*np.pi/180)

    if dot_product > thres_cos_1 or dot_product < thres_cos_2: # beside or opposite
        if np.abs(centroid1[0] - centroid2[0]) / 1024 > 0.7: # And different size of 2 objects (need implement)
            predicate = 'across' # '2'
            encode = '2'
            return (predicate, encode)
        else:
            predicate = 'near' # '29'
            encode = '29'
            return (predicate, encode)

    if dot_product <= thres_cos_1 and dot_product >= thres_cos_2: # below or above
        if box1_overlay < 0.1 and box2_overlay < 0.1:
            if centroid1[1] > centroid2[1] and box1[1] > centroid2[1]:
                predicate = 'under' # '43'
                encode = '43'
                return (predicate, encode)
            if centroid1[1] <= centroid2[1] and box1[3] <= centroid2[1]:
                predicate = 'above' # '1'
                encode = '1'
                return (predicate, encode)
            else:
                predicate = 'near' # '29'
                encode = '29'
                return (predicate, encode)
        else:
            predicate = 'near' # '29'
            encode = '29'
            return (predicate, encode)