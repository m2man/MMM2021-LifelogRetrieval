import SceneGraphParser.sng_parser as sng_parser
import sys
# THE FOLDER MYSCEAL_NLP_UTILS CONTAINS FUNTIONS EXTRACTING LOCATION AND DATE INFORMATION FROM A QUERY
from mysceal_nlp_utils.extract_info import process_query2


def pre_process_query(q):
    q = q.lower()
    if 'office' in q or 'workplace' in q:
        q += ' work'
        q += ' dublin'
        
    (tree_tags, (new_objects, split_keywords, region, locations, weekdays, start_time, end_time, dates)) = process_query2(q)
    discard_words = []
    print(tree_tags)
    for idx, tree in enumerate(tree_tags):
        if tree[1] in ['REGION', 'LOCATION', 'SPACE', 'TIME', 'WEEKDAY', 'DATE']:
            if tree_tags[idx-1][1] in ['IN', 'TO']:
                discard_words.append(f"{tree_tags[idx-1][0]} {tree[0]}")
            else:
                discard_words.append(tree[0])
    process_q = q
    for w in discard_words:
        process_q = process_q.replace(f" {w}", "")
    
    info = {}
    info['region'] = region
    info['location'] = locations
    info['weekday'] = weekdays
    info['start_time'] = start_time
    info['end_time'] = end_time
    info['date'] = dates
    
    return process_q, info
    
def generate_Graph_from_Text(text=''):
    graph = sng_parser.parse(text)
    list_sub = [x['head'] for x in graph['entities']]
    sgg = []
    for rel in graph['relations']:
        sub = rel['subject']
        obj = rel['object']
        pred = rel['lemma_relation']
        sub_str = f"{list_sub[sub]}:{sub}"
        obj_str = f"{list_sub[obj]}:{obj}"
        pair = [sub_str, pred, obj_str]
        sgg.append(pair)
   
    G=nx.DiGraph()
    for idx, sub in enumerate(list_sub):
        subj = f"{sub}:{idx}"
        G.add_node(subj)
        G.nodes[subj]['decode_obj'] = sub
        
    for edge in sgg:
        G.add_edge(edge[0], edge[2], decode_sub=edge[0].split(':')[0], decode_obj=edge[2].split(':')[0], decode_rel=edge[1])
    
    return G, sgg