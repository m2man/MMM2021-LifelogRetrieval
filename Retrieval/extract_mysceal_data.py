'''
THIS FILE IS TO EXTRACT LIFELOG INFORMATION OF EACH IMAGE IN LSC DATA
THIS INCLUDE TIME, DATE, LOCATION, AND OTHER DETECTED CONCEPTS
'''

import os
import json
import joblib
from datetime import datetime
from tqdm import tqdm
os.sys.path.append('/home/nmduy/Graph')

def extract_info_from_date(date_str):
    if '+00' in date_str: # remove the +00 at the end
        date_str = date_str[:-3]
    date_time = datetime.strptime(date_str, '%Y/%m/%d %H:%M:%S') 
    result = {}
    result['year'] = date_time.year
    result['month'] = date_time.month
    result['day'] = date_time.day
    result['hour'] = date_time.hour
    result['min'] = date_time.minute
    return result

def extract_data():
    COMMON_PATH = os.getenv('COMMON_PATH')

    with open(f'{COMMON_PATH}/grouped_info_dict.json') as f:
        data = json.load(f)

    bbox_1 = joblib.load('data_joblib/bbox_lsc2018.joblib')
    list_days = list(bbox_1.keys())
    list_processed_id = []
    for day in list_days:
        list_processed_id += list(bbox_1[day].keys())
    del bbox_1

    tags_keys = ['coco', 'deeplab', 'microsoft_tags', 'category', 'attributes']
    location_keys = ['address', 'location', 'region']

    result = {}

    for img_id in tqdm(list_processed_id):
        # img_id = list_processed_days[23324]
        data_img = data[img_id]
        ## time, weekday, coco, deeplab, microsoft_tags, category, attributes, location, region
        time = data_img['time']
        time_extract = extract_info_from_date(time)
        time_extract['weekday'] = data_img['weekday'].lower()

        tags = []
        for key in tags_keys:
            meta = data_img[key]
            meta = meta if isinstance(meta, list) else [meta]
            for words in meta:
                list_words = words.split(' ')
                list_words = [x.lower() for x in list_words]
                for word in list_words:
                    if word not in tags:
                        tags.append(word)
        location = []
        for key in location_keys:
            meta = data_img[key]
            meta = meta if isinstance(meta, list) else [meta]
            for words in meta:
                words = words.replace('(','')
                words = words.replace(')','')
                words = words.replace(',','')
                list_words = words.split(' ')
                list_words = [x.lower() for x in list_words]
                for word in list_words:
                    if word not in location and word not in ['', ' ', 'a', 'an', 'the']:
                        location.append(word)

        img_info = {}
        img_info['time'] = time_extract
        img_info['tags'] = tags
        img_info['location'] = location

        result[img_id] = img_info

    joblib.dump(result, 'mysceal_data.joblib')
    
def embed_mysceal():
    import graph_lib
    data = joblib.load('data_joblib/mysceal_data.joblib')
    list_id = list(data.keys())
    list_id = sorted(list_id)
    embed_matrix = {}
    time_dict = {}
    location_dict = {}
    for img_id in tqdm(list_id):
        data_img = data[img_id]
        #a = graph_lib.create_Graph_from_objects(data_img['tags'], idx_to_label=False)
        #node, pred = graph_lib.get_graph_embedding_concatenate(a)
        time_dict[img_id] = data_img['time']
        location_dict[img_id] = data_img['location']
        #embed_matrix[img_id] = {}
        #embed_matrix[img_id]['nodes'] = node
        #embed_matrix[img_id]['edges'] = pred
    
    joblib.dump(time_dict, 'mysceal_time.joblib')
    joblib.dump(location_dict, 'mysceal_location.joblib')
    #joblib.dump(embed_matrix, 'mysceal_score_matrix_concatenate.joblib')
    
print('START')
# extract_data()
embed_mysceal()
print('DONE')