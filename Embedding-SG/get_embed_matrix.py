import networkx as nx
import joblib
import matplotlib.pyplot as plt
import json
import requests
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import sys

import utils as ut
import vrg_lib
import graph_lib
import sgg_lib

with open('/home/nmduy/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-dicts-with-attri.json') as f:
    info_dict = json.load(f)
    
IMG_FOLDER = '/mnt/DATA/lsc2020/'

print('Loading joblib files ...')
sgg = joblib.load('data_joblib/sgg_lsc2018_p1.joblib')
bbox = joblib.load('data_joblib/bbox_lsc2018_p1.joblib')
print('Loading joblib files --> DONE!')
list_days = list(sgg.keys())
list_days = sorted(list_days)

# score_nodes = {}
# score_preds = {}
score = {}
sgg_dicts = {}
graph_dicts = {}

## Process all days
for idx_day in range(len(list_days)):
    sample_day = sgg[list_days[idx_day]]
    list_images = list(sample_day.keys())
    list_images = sorted(list_images)
    print(f"Processing folder {list_days[idx_day]}: {len(list_images)} images")
    
    ## Process all images in 1 day
    for idx_image in tqdm(range(len(list_images))):
        img_id = list_images[idx_image]
        sample_sgg = sample_day[img_id]
        sample_bbox = bbox[list_days[idx_day]][img_id]
        discard = False
        if len(sample_sgg) == 0:
            if len(sample_bbox) > 0 and len(sample_bbox['labels']) > 0:
                G_refill = graph_lib.create_Graph_from_objects(sample_bbox['labels'], sample_bbox['bbox'], info_dict)
            else:
                discard = True
                print(f"Discard images: {img_id}")
        else:        
            ## Filter lower score objects
            try:
                filter_sgg, filter_bbox = sgg_lib.filter_triplet(sample_sgg, sample_bbox, thres_obj=0.2, thres_rel=1e-4)
            except:
                print(f"[Filter error] idx_day: {idx_day}, idx_image: {idx_image}")
                print(f"Error:\n {sys.exc_info()}")
                exit() 

            ## Merge overlay objects
            try:
                overlay_sgg, overlay_bbox = sgg_lib.remove_overlay_objects(filter_sgg, filter_bbox)
            except:
                print(f"[Overlay error] idx_day: {idx_day}, idx_image: {idx_image}, img_id: {img_id}")
                print(f"Error:\n {sys.exc_info()}")
                exit() 

            ## Prune to have only 1 edge between 2 objects
            #prune_bbox = overlay_bbox.copy()
            try:
                prune_sgg = sgg_lib.prune_single_predicate_each_pair(overlay_sgg)
                G_prune = graph_lib.create_Graph(prune_sgg, overlay_bbox['bbox'], info_dict) # Create graph
            except:
                print(f"[Prune error] idx_day: {idx_day}, idx_image: {idx_image}, img_id: {img_id}")
                print(f"Error:\n {sys.exc_info()}")
                exit()

            ## Expand the spatial
            #expand_bbox = prune_bbox.copy()
            try:
                expand_sgg = sgg_lib.get_fully_connected_graph(prune_sgg, overlay_bbox)
            except:
                print(f"[VDG error] idx_day: {idx_day}, idx_image: {idx_image}, img_id: {img_id}")
                print(f"Error:\n {sys.exc_info()}")
                exit()

            ## Create Graph
            G_fc = graph_lib.create_Graph(expand_sgg, overlay_bbox['bbox'], info_dict)

            ## MST
            G_mst = nx.algorithms.tree.mst.maximum_spanning_tree(G=G_fc.to_undirected(), weight='score')

            ## Merge graphs: MST Graph + Prune Graph + priority for Prune Graph (keep_origin=False)
            G_merge, merge_sgg = graph_lib.merge_Graphs(G_mst, G_prune, keep_origin=False)

            ## Add object detected but no predicated (alone node)
            try:
                G_refill = graph_lib.fill_in_single_node_from_bbox(G_merge, overlay_bbox, info_dict)
            except:
                print(f"[Fill-in error] idx_day: {idx_day}, idx_image: {idx_image}, img_id: {img_id}")
                print(f"Error:\n {sys.exc_info()}")
                exit()
        
        if discard:
            node_matrix = []
            pred_matric = []
            merge_sgg = []
            overlay_bbox = []
            G_refill = None
        else:
            ## Embed to get score matrix
            node_matrix, pred_matrix = graph_lib.get_graph_embedding_concatenate(G_refill)
        
        ## Assign to dict
        # score_nodes[img_id] = node_matrix
        # score_preds[img_id] = pred_matrix
        score[img_id] = {}
        score[img_id]['nodes'] = node_matrix
        score[img_id]['edges'] = pred_matrix
        sgg_dicts[img_id] = {}
        sgg_dicts[img_id]['sgg'] = merge_sgg
        sgg_dicts[img_id]['bbox'] = overlay_bbox
        graph_dicts[img_id] = G_refill
        
print('Saving ...')
# joblib.dump(score_nodes, 'node_embed_lsc2018.joblib')
# joblib.dump(score_preds, 'pred_embed_concatenate_lsc2018.joblib')
joblib.dump(score, 'score_matrix_concatenate_lsc2018.joblib')
joblib.dump(sgg_dicts, 'sgg_mst_lsc2018.joblib')
joblib.dump(graph_dicts, 'image_graph_lsc2018.joblib')

