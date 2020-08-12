from PIL import Image
import torch
import numpy as np
import json
from maskrcnn_benchmark.data.transforms.build import my_build_transforms
from maskrcnn_benchmark.structures.image_list import *
from maskrcnn_benchmark.structures.bounding_box import BoxList
#torch.cuda.set_device(1)

with open('datasets/vg/VG-SGG-dicts-with-attri.json', 'r') as f:
    info_dict = json.load(f)

def refine_boxlist(boxlist, thres_obj=0.1, thres_rel=0.1):
    device = torch.device('cpu')
    
    size = boxlist.size
    mode = boxlist.mode
    
    bbox = boxlist.bbox.to(device)
    pred_scores = boxlist.get_field('pred_scores').to(device)
    pred_labels = boxlist.get_field('pred_labels').to(device)
    pair_obj = boxlist.get_field('rel_pair_idxs').to(device)
    rel_labels = boxlist.get_field('pred_rel_labels').to(device)
    rel_scores = boxlist.get_field('pred_rel_scores').to(device)
    
    rel_scores = rel_scores.numpy()
    rel_labels = rel_labels.numpy()
    pair_obj = pair_obj.numpy()
    pred_scores = pred_scores.numpy()
    pred_labels = pred_labels.numpy()
    bbox = bbox.numpy()

    # Remove detected object with lower score
    idx_low_pred_labels = np.where(pred_scores < thres_obj)[0]
    idx_high_pred_labels = [x for x in range(len(pred_scores)) if x not in list(idx_low_pred_labels)]
    
    # Remove detected relation corresponding with low score detected object
    idx_pairs_contain_low_labels = [idx for idx, row in enumerate(pair_obj) if row[0] in idx_low_pred_labels or row[1] in idx_low_pred_labels]
    idx_pairs_contain_high_labels = [x for x in range(len(rel_scores)) if x not in idx_pairs_contain_low_labels]
    
    pairs_contain_high_labels = pair_obj[idx_pairs_contain_high_labels,:]
    rel_labels_contain_high_labels = rel_labels[idx_pairs_contain_high_labels]
    rel_scores_contain_high_labels = rel_scores[idx_pairs_contain_high_labels,:]
    pred_labels_contain_high_labels = pred_labels[idx_high_pred_labels]
    pred_scores_contain_high_labels = pred_scores[idx_high_pred_labels]
    bbox_contain_high_labels = bbox[idx_high_pred_labels,:]
    
    # Reindex object in the pairs object
    temp = idx_high_pred_labels
    for i in range(pairs_contain_high_labels.shape[0]):
        row = pairs_contain_high_labels[i]
        pairs_contain_high_labels[i][0] = temp.index(row[0])
        pairs_contain_high_labels[i][1] = temp.index(row[1])
        
    # Should remove detected relation with lower score
    max_scores = np.max(rel_scores_contain_high_labels[:,1:], axis=1)
    idx_high_rel = np.where(max_scores > thres_rel)[0]
    pairs_contain_high_labels_high_rels = pairs_contain_high_labels[idx_high_rel,:]
    rel_labels_contain_high_labels_high_rels = rel_labels_contain_high_labels[idx_high_rel]
    rel_scores_contain_high_labels_high_rels = rel_scores_contain_high_labels[idx_high_rel,:]
    max_scores_filtered = max_scores[idx_high_rel]
    
    newboxlist = BoxList(bbox=bbox_contain_high_labels, image_size=size, mode=mode)
    newboxlist.add_field('pred_scores', pred_scores_contain_high_labels)
    newboxlist.add_field('pred_labels', pred_labels_contain_high_labels)
    newboxlist.add_field('pred_rel_labels', rel_labels_contain_high_labels_high_rels)
    newboxlist.add_field('pred_rel_scores', rel_scores_contain_high_labels_high_rels)
    newboxlist.add_field('rel_pair_idxs', pairs_contain_high_labels_high_rels)
    newboxlist.add_field('pred_rel_max_scores', max_scores_filtered)
    
    return newboxlist

def translate_to_human_read(decode_matrix_list, info_dict=info_dict):
    result = []
    for i in range(len(decode_matrix_list)):
        line_rel = decode_matrix_list[i]
        obj1 = line_rel[0].split(':')
        obj2 = line_rel[2].split(':')
        rel = line_rel[1]
        obj1 = info_dict['idx_to_label'][str(obj1[0])] + ':' + str(obj1[1])
        obj2 = info_dict['idx_to_label'][str(obj2[0])] + ':' + str(obj2[1])
        con = info_dict['idx_to_predicate'][str(rel)]
        if len(line_rel) > 3:
            score = line_rel[3]
            result.append([obj1, con, obj2, score])
        else:
            result.append([obj1, con, obj2])
    return result

def decode_relation(boxlist, show_scores=False):
    # show_score will show the sgg_score: average score for sub - pred - obj
    pair_array = boxlist.get_field('rel_pair_idxs')
    rel_label_array = boxlist.get_field('pred_rel_labels')
    label_array = boxlist.get_field('pred_labels')
    try:
        sgg_score_array = boxlist.get_field('sgg_scores')
    except:
        show_scores=False
        print("Cannot find attribute sgg_scores")

    result = []
    for i in range(pair_array.shape[0]):
        rel = pair_array[i]
        obj1 = str(label_array[rel[0]]) + ':' + str(rel[0])
        obj2 = str(label_array[rel[1]]) + ':' + str(rel[1])
        con = str(rel_label_array[i])
        if show_scores:
            score = sgg_score_array[i]
            result.append([obj1, con, obj2, score])
        else:
            result.append([obj1, con, obj2])
    return result

def ranking_boxlist(boxlist):
    # This function is to reranking the order of the sgg scores by 0.2*score_obj_1 + 0.4*pred + 0.2*score_obj_2
    rel_scores_value = boxlist.get_field('pred_rel_max_scores')
    pred_scores = boxlist.get_field('pred_scores')
    relation_pair = boxlist.get_field('rel_pair_idxs')
    sgg_scores = np.zeros(relation_pair.shape[0])
    for i in range(relation_pair.shape[0]):
        pair = relation_pair[i]
        score = 0.2*pred_scores[pair[0]] + 0.2*pred_scores[pair[1]] + 0.4*rel_scores_value[i]
        sgg_scores[i] = score
    idx_descending = sgg_scores.argsort()[::-1]
    
    newboxlist = BoxList(bbox=boxlist.bbox, image_size=boxlist.size, mode=boxlist.mode)
    newboxlist.add_field('pred_scores', boxlist.get_field('pred_scores'))
    newboxlist.add_field('pred_labels', boxlist.get_field('pred_labels'))
    newboxlist.add_field('pred_rel_labels', boxlist.get_field('pred_rel_labels')[idx_descending])
    newboxlist.add_field('pred_rel_scores', boxlist.get_field('pred_rel_scores')[idx_descending])
    newboxlist.add_field('rel_pair_idxs', relation_pair[idx_descending])
    newboxlist.add_field('pred_rel_max_scores', rel_scores_value[idx_descending])
    newboxlist.add_field('sgg_scores', sgg_scores[idx_descending])
    return newboxlist

