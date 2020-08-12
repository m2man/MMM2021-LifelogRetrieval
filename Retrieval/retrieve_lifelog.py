import os
import joblib
import json
import numpy as np
os.sys.path.append('./Embedding-SG')
import graph_lib
os.sys.path.append('./Parsing-Query')
import nlp_lib
import multiprocessing as mp

DATA_FOLDER = '/mnt/DATA/nmduy/MMM2021/graph_matrix'
IMAGE_FOLDER = '/mnt/DATA/lsc2020'

mysceal_location = joblib.load(f'{DATA_FOLDER}/mysceal_location.joblib')
mysceal_time = joblib.load(f'{DATA_FOLDER}/mysceal_time.joblib')

embed_matrix = joblib.load(f'{DATA_FOLDER}/score_matrix_concatenate_lsc2018.joblib')
images_info_dict = joblib.load(f"{DATA_FOLDER}/sgg_mst_lsc2018.joblib")

with open('/home/nmduy/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-dicts-with-attri.json') as f:
    info_dict = json.load(f)

list_days = sorted(list(embed_matrix.keys()))

def find_similar_singlecore(query_score, embed_matrix, alpha=1, beta=1):
    score = np.zeros(len(embed_matrix))
    list_id = sorted(list(embed_matrix.keys()))
    
    for idx in range(len(score)):
        score_node, score_pred = graph_lib.similar_score(embed_matrix[list_id[idx]], query_score)
        score[idx] = alpha*score_node + beta*score_pred
        
    return score

def time_filter_result(initial_id_img_ranking, initial_score_ranking, query_info, mysceal_time_metadata):
    query_start_time = query_info['start_time'][0]
    query_end_time = query_info['end_time'][0]
    query_date = query_info['date']
    query_weekday = query_info['weekday']
    
    discard_index = []
    
    if query_start_time == 0 and query_end_time == 24 and len(query_weekday) == 0 and len(query_date) == 0: # DO NOT THING
        return initial_id_img_ranking, initial_score_ranking
    
    for idx, img_id in enumerate(initial_id_img_ranking):
        if query_start_time > 0:
            # process start time filter
            if mysceal_time_metadata[img_id]['hour'] < query_start_time:
                discard_index.append(idx)
                continue

        if query_end_time < 24:
            # process end_time filter
            if mysceal_time_metadata[img_id]['hour'] > query_end_time:
                discard_index.append(idx)
                continue

        if len(query_weekday) > 0:
            # process filter weekday
            if mysceal_time_metadata[img_id]['weekday'] not in query_weekday:
                discard_index.append(idx)
                continue

        if len(query_date) > 0:
            # process filter date
            date = query_date[0]
            year, month, day = date
            if year is not None: 
                if mysceal_time_metadata[img_id]['year'] != year :
                    discard_index.append(idx)
                    continue
            if month is not None: 
                if mysceal_time_metadata[img_id]['month'] != month :
                    discard_index.append(idx)
                    continue
            if day is not None: 
                if mysceal_time_metadata[img_id]['day'] != day :
                    discard_index.append(idx)
                    continue

    keep_index = [x for x in range(len(initial_id_img_ranking)) if x not in discard_index]

    id_img_time_filter = [initial_id_img_ranking[x] for x in keep_index]
    score_img_time_filter = initial_score_ranking[keep_index]
    
    return id_img_time_filter, score_img_time_filter

def location_filter_result(id_img_time_filter, score_time_filter, query_info, mysceal_location_metadata):
    ## LOCATION FILTER
    query_location = query_info['region']+query_info['location']
    inc_location = 0.1
    ini_bonus = 5
    inc_list = [1 for x in range(len(id_img_time_filter))]
    
    if len(query_location) == 0:
        return id_img_time_filter, score_time_filter
    
    for idx_img, img_id in enumerate(id_img_time_filter):
        increment = 0
        if len(query_location) > 0:
            # process location filter
            count = 0
            total = 0
            for idx, que_loc in enumerate(query_location):
                word = que_loc.split(' ')
                word = [x for x in word if x not in ['a', 'the']]
                for w in word:
                    total += 1
                    if w in mysceal_location_metadata[img_id]:
                        count += 1

            increment = increment + count / total * inc_location
        
        inc_list[idx_img] = (1+increment)

    inc_value = np.asarray(inc_list)
    score_location_filter = (score_time_filter+ini_bonus) * inc_value
    idx_ranking = np.argsort(score_location_filter)[::-1].tolist()
    score_location_filter = score_location_filter[idx_ranking]
    id_img_location_filter = [id_img_time_filter[x] for x in idx_ranking]
    
    return id_img_location_filter, score_location_filter


def time_filter_each_image(idx, mysceal_time_img, query_info):
    score = 0 # 0 is ok , 1 is filtered out
    
    query_start_time = query_info['start_time'][0]
    query_end_time = query_info['end_time'][0]
    query_date = query_info['date']
    query_weekday = query_info['weekday']
    
    if query_start_time > 0:
        # process start time filter
        if mysceal_time_img['hour'] < query_start_time:
            score = 1
            return (idx, score)

    if query_end_time < 24:
        # process end_time filter
        if mysceal_time_img['hour'] > query_end_time:
            score = 1
            return (idx, score)

    if len(query_weekday) > 0:
        # process filter weekday
        if mysceal_time_img['weekday'] not in query_weekday:
            score = 1
            return (idx, score)

    if len(query_date) > 0:
        # process filter date
        date = query_date[0]
        year, month, day = date
        if year is not None: 
            if mysceal_time_img['year'] != year :
                score = 1
                return (idx, score)
        if month is not None: 
            if mysceal_time_img['month'] != month :
                score = 1
                return (idx, score)
        if day is not None: 
            if mysceal_time_img['day'] != day :
                score = 1
                return (idx, score)
            
    return (idx, score)

def time_filter_result_multicore(initial_id_img_ranking, initial_score_ranking, query_info, mysceal_time_metadata, numb_core=None):
    query_start_time = query_info['start_time'][0]
    query_end_time = query_info['end_time'][0]
    query_date = query_info['date']
    query_weekday = query_info['weekday']
    
    if query_start_time == 0 and query_end_time == 24 and len(query_weekday) == 0 and len(query_date) == 0: # DO NOTHING
        return initial_id_img_ranking, initial_score_ranking
    
    # MULTIPROCESSING 
    if numb_core is None:
        pool = mp.Pool(mp.cpu_count())
    else:
        pool = mp.Pool(numb_core)
    # call apply_async() without callback
    result_objects = [pool.apply_async(time_filter_each_image, args=(i, mysceal_time_metadata[img_id], query_info)) for i, img_id in enumerate(initial_id_img_ranking)]
    # result_objects is a list of pool.ApplyResult objects
    filtered_out = [r.get()[1] for r in result_objects]
    pool.close()
    pool.join()
    
    keep_index = [idx for idx, x in enumerate(filtered_out) if x == 0]
    
    id_img_time_filter = [initial_id_img_ranking[x] for x in keep_index]
    score_img_time_filter = initial_score_ranking[keep_index]
    
    return id_img_time_filter, score_img_time_filter


def retrieve_lifelog(query='', numb=100):
    thres_initial = 2000
    
    print(f"===== INPUT =====\nQuery:{query}\nNumb_Imgs:{numb}\n=====================") 
    
    processed_query, query_info = nlp_lib.pre_process_query(query)
    
    print(f"----- Processed Query -----\n{processed_query}\n----------------------")
    print(f"----- Query Info -----\n{query_info}\n----------------------")
    
    G_text, text_sgg = graph_lib.generate_Graph_from_Text(text=processed_query)
    
    text_sgg_dict = graph_lib.convert_sgg_to_dict(text_sgg, list(G_text.nodes()))
    
    # FOR FORCE GRAPH
    text_sgg_dict = graph_lib.convert_dict_to_forcegraph_format(text_sgg_dict)
    
    node_matrix, pred_matrix = graph_lib.get_graph_embedding_concatenate(G_text)
    query_score = {}
    query_score['nodes'] = node_matrix
    query_score['edges'] = pred_matrix
    
    start = time.time()
    
    print('Searching ...')
    initial_score = find_similar_singlecore(query_score, embed_matrix, alpha=1, beta=1.7)
    idx_ranking = np.argsort(initial_score)[::-1].tolist()
    initial_score = initial_score[idx_ranking]
    id_img_ranking = [list_days[x] for x in idx_ranking]
    
    initial_score = initial_score[:thres_initial]
    id_img_ranking = id_img_ranking[:thres_initial]
    
    end_search = time.time()
    
    print('Time Filtering ...')
    #id_img_time_filter, score_time_filter = time_filter_result(id_img_ranking, initial_score, query_info, mysceal_time)
    id_img_time_filter, score_time_filter = time_filter_result_multicore(id_img_ranking, initial_score, query_info, mysceal_time)
    
    end_time_filter = time.time()
    
    print('Location Filtering ...')
    id_img_location_filter, score_location_filter = location_filter_result(id_img_time_filter, score_time_filter, query_info, mysceal_location)
    
    end_location_filter = time.time()
    
    print(f'----- Finish scoring -----\nInitial Search: {round(end_search-start, 2)} sec\nTime Filter: {round(end_time_filter-end_search, 2)} sec\nLocation Filter: {round(end_location_filter-end_time_filter, 2)} sec\nTotal Search: {round(end_location_filter-start, 2)} sec\n-------------------')
    
    
    if numb:
        numb = min(int(numb), len(id_img_location_filter))
    else:
        numb = len(id_img_location_filter)
    
    result = [id_img_location_filter[x][:-4]+'.webp' for x in range(numb)]
    
    response = {'result': result, 'text_sgg_dict': text_sgg_dict}
    
    
##### RUN #####
query = 'a man was holding a cup'
result = retrieve_lifelog(query=query, numb=100)