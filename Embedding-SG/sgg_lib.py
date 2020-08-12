import numpy as np
import vrg_lib 
import utils as ut

def prune_single_predicate_each_pair(original_sgg):
    '''
    original_sgg = sample_sgg
    input is the original sgg from the server
    output is smaller sgg
    '''
    prune_sgg = []
    list_pair = []
    list_idx_pair = []
    for idx_pair, pair in enumerate(original_sgg):
        obj_pair = [pair[0], pair[2]]
        reverse = obj_pair[::-1]
        if obj_pair not in list_pair and reverse not in list_pair:
            list_pair.append(obj_pair)
            list_idx_pair.append(idx_pair)
        else:
            try:
                current_score = original_sgg[list_idx_pair[list_pair.index(obj_pair)]][3]
            except:
                current_score = original_sgg[list_idx_pair[list_pair.index(reverse)]][3]
            new_score = pair[3]
            if new_score > current_score:
                list_idx_pair[list_pair.index(pair)] = idx_pair
    for idx in list_idx_pair:
        prune_sgg.append(original_sgg[idx])
    
    return prune_sgg

def get_rel_scores(sample_sgg, bbox_scores):
    '''
    GET THE SCORES OF EACH PREDICATES
    sample_sgg is list of list with format [[sub, pred, obj], [...]]
    bbox_scores is list of scores for each bbox in sub, obj with format [ ... ]
    '''
    rel_scores = []
    for sgg in sample_sgg:
        sub, pred, obj, overall_score = sgg[0], sgg[1], sgg[2], sgg[3]
        idx_sub = int(sub.split(':')[-1])
        idx_obj = int(obj.split(':')[-1])
        sub_score = bbox_scores[idx_sub]
        obj_score = bbox_scores[idx_obj]
        pred_score = (overall_score - 0.2 * (sub_score + obj_score))/0.4
        triple = [sub, pred, obj, pred_score]
        rel_scores.append(triple)
    return rel_scores

def filter_triplet(input_sgg, bbox_dict, thres_obj=0.1, thres_rel=0.1):
    '''
    FILTER OUT SOME LOW SCORE OBJ AND CORRESSPONDING PREDICATE
    sample_sgg is list of list with format [[sub, pred, obj], [...]]
    bbox_out is dictionary keys bbox, labels, scores
    '''
    
    sample_sgg = input_sgg.copy()
    if len(bbox_dict) == 0:
        return input_sgg, bbox_dict # no object detection
    
    bbox_scores = bbox_dict['scores']
    bbox_labels = bbox_dict['labels']
    bbox = bbox_dict['bbox']
    
    bbox_scores_np = np.asarray(bbox_scores)
    idx_high_pred_labels = np.where(bbox_scores_np > thres_obj)[0]
    
    idx_pairs_contain_high_labels = [idx for idx, row in enumerate(sample_sgg) if int(row[0].split(':')[-1]) in idx_high_pred_labels and int(row[2].split(':')[-1]) in idx_high_pred_labels]
        
    sgg_high = [sample_sgg[x] for x in idx_pairs_contain_high_labels]
    bbox_scores_high = [bbox_scores[x] for x in idx_high_pred_labels]
    bbox_high = [bbox[x] for x in idx_high_pred_labels]
    bbox_labels_high = [bbox_labels[x] for x in idx_high_pred_labels]
    
    # Reindex object in the pairs object
    temp = idx_high_pred_labels.tolist()
    
    for i in range(len(sgg_high)):
        row = sgg_high[i]
        sub, pred, obj, overall_score = row[0], row[1], row[2], row[3]
        sub = f"{sub.split(':')[0]}:{temp.index(int(sub.split(':')[1]))}"
        obj = f"{obj.split(':')[0]}:{temp.index(int(obj.split(':')[1]))}"
        if pred == '30': # convert 'of' to 'has'
            pred = '20'
            temp_exchange = sub
            sub = obj
            obj = temp_exchange 
        sgg_high[i] = [sub, pred, obj, overall_score]
        
    # Should remove detected relation with lower score
    rel_scores = get_rel_scores(sgg_high, bbox_scores_high)
    max_scores = np.asarray([x[3] for x in rel_scores])
    
    idx_high_rel = np.where(max_scores > thres_rel)[0]
    sgg_high_rel = [sgg_high[x] for x in idx_high_rel]
    
    bbox_dict_high = {'bbox': bbox_high, 'labels': bbox_labels_high, 'scores': bbox_scores_high}
    
    return sgg_high_rel, bbox_dict_high

def get_fully_connected_graph(sample_sgg, sample_bbox_dict):
    '''
    input: bbox, bbox scores, sgg
    bbox to classify the spatial relationship
    scores to priority with bbox should be expand first, as there only 1 edge between nodes
    sgg to keep the original predicate
    output: expanded sgg
    '''
    if len(sample_bbox_dict) == 0:
        print('WARNING: BBOX DICT HAS NO LENGTH')
        return sample_sgg
    input_sgg = sample_sgg.copy()
    input_bbox = sample_bbox_dict['bbox']
    input_bbox_labels = sample_bbox_dict['labels']
    input_bbox_scores = sample_bbox_dict['scores']

    idx_sort_score = np.argsort(np.asarray(input_bbox_scores))[::-1]

    list_triplets = input_sgg

    list_pair = [[int(x[0].split(':')[1]), int(x[2].split(':')[1])] for x in input_sgg]
    for idx_idx_sub in range(len(idx_sort_score)-1):
        idx_sub = idx_sort_score[idx_idx_sub]
        for idx_idx_obj in range(idx_idx_sub+1, len(idx_sort_score)):
            idx_obj = idx_sort_score[idx_idx_obj]
            pair = [idx_sub, idx_obj]
            reverse = pair[::-1]
            if pair in list_pair or reverse in list_pair:
                continue
            else:
                # perform the VDG
                bb1 = input_bbox[idx_sub]
                bb2 = input_bbox[idx_obj]
                predicate = vrg_lib.get_VDG(bb1, bb2)
                score = 0.1 * (input_bbox_scores[idx_sub] + input_bbox_scores[idx_obj])
                sub = f"{input_bbox_labels[idx_sub]}:{idx_sub}"
                obj = f"{input_bbox_labels[idx_obj]}:{idx_obj}"
                triplet = [sub, predicate[1], obj, score]
                list_triplets.append(triplet)
                
    return list_triplets

def remove_overlay_objects(sample_sgg, sample_bbox):
    '''
    Remove some object detected in the same area due to the limitation of object detection algo in the SGG
    Alter the relationship of overlay objects
    '''
    same_objects = [90, 91, 68, 20, 53, 78, 79, 70, 29, 56, 149]
    overlap_matrix = np.ones((len(sample_bbox['scores']), len(sample_bbox['scores'])))
    for i in range(overlap_matrix.shape[0]):
        for j in range(overlap_matrix.shape[0]):
            overlap_matrix[i,j] = np.round(ut.percent_overlap(sample_bbox['bbox'][i], sample_bbox['bbox'][j]), 2)

    overlap_bbox = []
    for i in range(overlap_matrix.shape[0]):
        for j in range(overlap_matrix.shape[0]):
            box_i_label = sample_bbox['labels'][i]
            box_j_label = sample_bbox['labels'][j]
            box_i = sample_bbox['bbox'][i]
            box_j = sample_bbox['bbox'][j]
            box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            cri_1 = overlap_matrix[i,j] < 0.99 and overlap_matrix[i,j] > 0.65
            cri_2 = overlap_matrix[j,i] < 0.99 and overlap_matrix[j,i] > 0.65
            cri_3 = box_i_label in same_objects and box_j_label in same_objects
            cri_4 = box_i_area/box_j_area > 0.7 and box_i_area/box_j_area < 1/0.7
            if cri_1 and cri_2 and cri_3 and cri_4:
                if [i, j] not in overlap_bbox and [j, i] not in overlap_bbox:
                    if sample_bbox['scores'][i] > sample_bbox['scores'][j]:
                        overlap_bbox.append([i,j])
                    else:
                        overlap_bbox.append([j,i])
                        
    discard_box = [x[1] for x in overlap_bbox]
    alter_box_temp = [x[0] for x in overlap_bbox]
    alter_box = []
    # Sometimes many objects are overlayed
    for k in alter_box_temp:
        while k in discard_box:
            idx_discard = discard_box.index(k)
            k = alter_box_temp[idx_discard]
        alter_box.append(k)
        
    keep_box = [x for x in range(len(sample_bbox['bbox'])) if x not in discard_box]
    new_bbox_labels = [sample_bbox['labels'][x] for x in keep_box]
    new_bbox_scores= [sample_bbox['scores'][x] for x in keep_box]
    new_bbox = [sample_bbox['bbox'][x] for  x in keep_box]

    sgg_out = []
    for rel in sample_sgg:
        sub, pred, obj, score = rel 
        sub_label, sub_idx =  sub.split(':')
        obj_label, obj_idx =  obj.split(':')

        if int(sub_idx) in discard_box:
            idx_discard = discard_box.index(int(sub_idx))
            sub_idx = str(keep_box.index(alter_box[idx_discard]))
            sub_label = str(new_bbox_labels[int(sub_idx)])
        else:
            sub_idx = str(keep_box.index(int(sub_idx)))

        if int(obj_idx) in discard_box:
            idx_discard = discard_box.index(int(obj_idx))
            obj_idx = str(keep_box.index(alter_box[idx_discard]))
            obj_label = str(new_bbox_labels[int(obj_idx)])
        else:
            obj_idx = str(keep_box.index(int(obj_idx)))

        if sub_idx != obj_idx: # if there is a predicate between same objects --> discard
            sgg_out.append([f"{sub_label}:{sub_idx}", pred, f"{obj_label}:{obj_idx}", score])

    bbox_out = {'bbox': new_bbox, 'labels': new_bbox_labels, 'scores': new_bbox_scores}
    
    return sgg_out, bbox_out