import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import numpy as np
from nltk.stem import WordNetLemmatizer 

W2V_MODEL_PATH = '/mnt/DATA/nmduy/GoogleNews-vectors-negative300.bin'
print('Loading word2vec model in graph_lib')
w2v_model = KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=True)
print('Loading word2vec model in graph_lib --> DONE!')
lemmatizer = WordNetLemmatizer()

def convert_nodes_to_human_read(list_nodes, info_dict):
    try:
        encode_nodes = [x.split(':')[0] for x in list_nodes]
    except:
        encode_nodes = [x for x in list_nodes]
    
    try:
        index_nodes = [x.split(':')[1] for x in list_nodes]
    except:
        index_nodes = None
    
    if index_nodes:
        decode_nodes = [info_dict['idx_to_label'][x]+':'+y for x,y in zip(encode_nodes, index_nodes)]
    else:
        decode_nodes = [info_dict['idx_to_label'][x] for x in encode_nodes]
        
    return decode_nodes

def convert_sgg_to_human_read(sgg, info_dict):
    result = []
    for i in range(len(sgg)):
        line_rel = sgg[i]
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

def draw_Graph(G, show_predicate=True):
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.circular_layout(G)
    nx.draw(G, pos, node_color='#ffeb0f')
    node_labels = nx.get_node_attributes(G, 'decode_obj')
    nx.draw_networkx_labels(G, pos, node_labels)
    if show_predicate:
        edge_labels = nx.get_edge_attributes(G,'decode_rel')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
def create_Graph(sgg, bbox=None, info_dict=None):
    G=nx.DiGraph()
    list_nodes = []
    obj_dict = {}
    list_edges = []
    list_rels = []
    for rel in sgg:
        for idx in [0, 2]:
            obj_name = rel[idx].split(':')[0]
            obj_idx = rel[idx].split(':')[1]
            if obj_name not in list(obj_dict.keys()):
                obj_dict[obj_name] = [obj_idx]
            else:
                obj_dict[obj_name].append(obj_idx)
                
            if rel[idx] not in list_nodes:
                list_nodes.append(rel[idx])
                obj = rel[idx]
                a = obj.split(':')[0]
                idx_bbox = int(obj.split(':')[1])
                b = info_dict['idx_to_label'][a]
                G.add_node(obj)
                G.nodes[obj]['encode_obj'] = a+':'+str(idx_bbox)
                G.nodes[obj]['decode_obj'] = b+':'+str(idx_bbox)
                if bbox:
                    G.nodes[obj]['bbox'] = bbox[idx_bbox]
        list_edges.append((rel[0], rel[2]))
        list_rels.append(rel[1])
        G.add_edge(rel[0], rel[2], decode_sub=info_dict['idx_to_label'][rel[0].split(':')[0]], decode_obj=info_dict['idx_to_label'][rel[2].split(':')[0]], encode_rel=rel[1], decode_rel=info_dict['idx_to_predicate'][rel[1]], score=rel[3])
    return G

def create_Graph_from_objects(objects, bbox=None, info_dict=None, idx_to_label=True):
    '''
    Create graph when there is no sgg support --> only nodes, no edges
    idx_to_label = True --> objects is encoded number (False --> obj is decoded label)
    '''
    G=nx.DiGraph()
    obj_dict = {}
    for idx, obj in enumerate(objects):
        obj = str(obj)
        name = f"{obj}:{idx}"
        G.add_node(name)
        if idx_to_label:
            G.nodes[name]['encode_obj'] = obj
        else:
            G.nodes[name]['decode_obj'] = obj
        if info_dict:
            if idx_to_label:
                G.nodes[name]['decode_obj'] = info_dict['idx_to_label'][obj]
            else:
                G.nodes[name]['encode_obj'] = info_dict['label_to_idx'][obj]
        if bbox:
            G.nodes[name]['bbox'] = bbox[idx]
    return G

def merge_Graphs(G1, G2, keep_origin=True):
    '''
    Bring all edges in G2 (but not appear in G1) to G1
    Should check the appearance of nodes in G2 and G1. If edges contain nodes in G2 but not in G1 --> dont add to G1
    keep_origin: Keep the edge on G1 if there is a duplicate edge on G2
    Also produce the SGG of new Graph G3
    '''
    sgg = []
    G=nx.DiGraph()
    list_nodes_G = []
    list_edges_G = []
    
    list_nodes_G1 = list(G1.nodes(data=True))
    list_nodes_G2 = list(G2.nodes(data=True))
    list_edges_G1 = list(G1.edges(data=True))
    list_edges_G2 = list(G2.edges(data=True))
    
    for node in list_nodes_G1:
        G.add_node(node[0])
        G.nodes[node[0]]['encode_obj'] = node[1]['encode_obj']
        G.nodes[node[0]]['decode_obj'] = node[1]['decode_obj']
        G.nodes[node[0]]['bbox'] = node[1]['bbox']
        list_nodes_G.append(node[0])
        
    for edge in list_edges_G1:
        G.add_edge(edge[0], edge[1], decode_sub=edge[2]['decode_sub'], decode_obj=edge[2]['decode_obj'], encode_rel=edge[2]['encode_rel'], decode_rel=edge[2]['decode_rel'], score=edge[2]['score'])
        list_edges_G.append([edge[0], edge[1]])
    
    for edge in list_edges_G2:
        if edge[0] in list_nodes_G and edge[1] in list_nodes_G:
            pair = [edge[0], edge[1]]
            if pair not in list_edges_G and pair[::-1] not in list_edges_G:
                G.add_edge(edge[0], edge[1], decode_sub=edge[2]['decode_sub'], decode_obj=edge[2]['decode_obj'], encode_rel=edge[2]['encode_rel'], decode_rel=edge[2]['decode_rel'], score=edge[2]['score'])
                list_edges_G.append([edge[0], edge[1]])
            else:
                if keep_origin:
                    continue
                else:
                    try:
                        G.remove_edge(pair[0], pair[1])
                    except:
                        G.remove_edge(pair[1], pair[0])
                    G.add_edge(edge[0], edge[1], decode_sub=edge[2]['decode_sub'], decode_obj=edge[2]['decode_obj'], encode_rel=edge[2]['encode_rel'], decode_rel=edge[2]['decode_rel'], score=edge[2]['score'])

    list_edges_G = list(G.edges(data=True))
    for edge in list_edges_G:
        sgg.append([edge[0], edge[2]['encode_rel'], edge[1], edge[2]['score']])
        
    return G, sgg
            
def fill_in_single_node_from_bbox(G, BBox, info_dict):
    '''
    add detected object in BBox (dictionary) but no predicate along with it
    BBox: dictionary with keys: bbox, labels, scores
    '''
    if len(BBox) == 0:
        return G
    nodes = list(G.nodes()) # nodes is in format of '123:4' (labels:index in bbox)
    index_nodes = [int(x.split(':')[1]) for x in nodes]
    add_nodes = [x for x in range(len(BBox['labels'])) if x not in index_nodes] # index of nodes in bbox but not in G
    for idx in add_nodes:
        encode_label = BBox['labels'][idx]
        node_label = f"{encode_label}:{idx}"
        decode_label = info_dict['idx_to_label'][str(encode_label)]
        G.add_node(node_label)
        G.nodes[node_label]['encode_obj'] = f"{encode_label}:{idx}"
        G.nodes[node_label]['decode_obj'] = f"{decode_label}:{idx}"
        G.nodes[node_label]['bbox'] = BBox['bbox'][idx]
    return G

def word2vec_text(t):
    # Get embedding of sequence of word by taking the average of all words in the text t
    w2v_size = 300
    t_split = t.split(' ')
    t_embed = np.zeros(w2v_size)
    count = 0
    for t_word in t_split:
        try:
            t_embed += w2v_model[t_word]
            count += 1
        except KeyError:
            file1 = open("KeyError_Word2Vec.txt","a") 
            file1.write(f'{t_word}\n')
            file1.close() 
    if count != 0:
        t_embed /= count
    else:
        t_embed = None
    return t_embed

def get_graph_embedding(G):
    '''
    Embed the nodes and predicates into matrices (1 for node, 1 for predicate)
    '''
    w2v_size = 300
    list_nodes_G = list(G.nodes(data=True))
    list_edges_G = list(G.edges(data=True))
    numb_nodes = len(list_nodes_G)
    numb_edges = len(list_edges_G)
    matrix_obj = np.zeros((0, w2v_size))
    matrix_rel = np.zeros((0, w2v_size))
    
    for idx_obj in range(numb_nodes):
        obj_label = list_nodes_G[idx_obj][1]['decode_obj'].split(':')[0]
        obj_label = lemmatizer.lemmatize(obj_label, 'n')
        embed = word2vec_text(obj_label)
        if embed is not None:
            embed = embed.reshape((1, w2v_size))
            matrix_obj = np.concatenate((matrix_obj, embed))
        
    for idx_rel in range(numb_edges):
        edge = list_edges_G[idx_rel]
        sub, pred, obj = edge[2]['decode_sub'], edge[2]['decode_rel'], edge[2]['decode_obj']
        if pred == 'next to':
            pred = 'beside'
        sub = lemmatizer.lemmatize(sub, 'n')
        pred = lemmatizer.lemmatize(pred, 'v')
        obj = lemmatizer.lemmatize(obj, 'n')
        
        sub_embed = word2vec_text(sub)
        pred_embed = word2vec_text(pred)
        obj_embed = word2vec_text(obj)
        
        if sub_embed is None:
            sub_embed = np.zeros(w2v_size)
        if pred_embed is None:
            pred_embed = np.zeros(w2v_size)
        if obj_embed is None:
            obj_embed = np.zeros(w2v_size)
            
        total_embed = 0.25*sub_embed + 0.5*pred_embed + 0.25*obj_embed
        matrix_rel[idx_rel] = total_embed
        
    return matrix_obj, matrix_rel

def get_graph_embedding_concatenate(G):
    '''
    Embed the nodes and predicates into matrices (1 for node, 1 for predicate)
    Difference with get_graph_embedding: now the sub-pred-obj will be concate --> 900-d vector
    '''
    w2v_size = 300
    list_nodes_G = list(G.nodes(data=True))
    list_edges_G = list(G.edges(data=True))
    numb_nodes = len(list_nodes_G)
    numb_edges = len(list_edges_G)
    matrix_obj = np.zeros((0, w2v_size))
    matrix_rel = np.zeros((numb_edges, w2v_size * 3)) # concatenate sub-pred-obj
    
    for idx_obj in range(numb_nodes):
        obj_label = list_nodes_G[idx_obj][1]['decode_obj'].split(':')[0]
        obj_label = lemmatizer.lemmatize(obj_label, 'n')
        embed = word2vec_text(obj_label)
        if embed is not None:
            embed = embed.reshape((1, w2v_size))
            matrix_obj = np.concatenate((matrix_obj, embed))
        
    for idx_rel in range(numb_edges):
        edge = list_edges_G[idx_rel]
        sub, pred, obj = edge[2]['decode_sub'], edge[2]['decode_rel'], edge[2]['decode_obj']
        if pred == 'next to':
            pred = 'beside'
        sub = lemmatizer.lemmatize(sub, 'n')
        pred = lemmatizer.lemmatize(pred, 'v')
        obj = lemmatizer.lemmatize(obj, 'n')
        
        sub_embed = word2vec_text(sub)
        pred_embed = word2vec_text(pred)
        obj_embed = word2vec_text(obj)
        
        if sub_embed is None:
            sub_embed = np.zeros(w2v_size)
        if pred_embed is None:
            pred_embed = np.zeros(w2v_size)
        if obj_embed is None:
            obj_embed = np.zeros(w2v_size)
            
        total_embed = np.concatenate((sub_embed, pred_embed, obj_embed))
        matrix_rel[idx_rel] = total_embed
        
    return matrix_obj, matrix_rel

def similar_score(V_dict, T_dict):
    '''
    Get the similarity score between 2 graphs
    input: V_dict is dictionary with matrix_obj, matrix_rel from Visual graph as above function
    input: T_dict is dictionary with matrix_obj, matrix_rel from Textual graph as above function
    '''
    V_nodes = V_dict['nodes']
    V_edges = V_dict['edges']
    T_nodes = T_dict['nodes']
    T_edges = T_dict['edges']
    
    if len(V_nodes) == 0 or len(T_nodes) == 0:
        S_nodes = 0
    else:
        Nodes = np.dot(T_nodes, V_nodes.T)
        S_nodes = np.mean(np.max(Nodes, axis=1))
    
    if len(V_edges) == 0 or len(T_edges) == 0:
        S_edges = 0
    else:
        Edges = np.dot(T_edges, V_edges.T)
        S_edges = np.mean(np.max(Edges, axis=1))
    
    # S = alpha*S_nodes + beta*S_edges
    
    return S_nodes, S_edges

# def convert_sgg_to_dict(sgg):
#     '''
#     Convert an sgg list of list into list of dict
#     Used for the Reactjs
#     ''' 
#     list_pred = []
#     for rel in sgg:
#         if len(rel) == 3:
#             sub, pred, obj = rel
#         else:
#             sub, pred, obj,_ = rel
#         rel_dict = {'sub': sub, 'pred': pred, 'obj': obj}
#         list_pred.append(rel_dict)
#     return list_pred

def convert_sgg_to_dict(sgg, list_nodes=None):
    '''
    Convert an sgg list of list dictionary of objects and predicates with its corresponding nodes
    Used for the Reactjs
    ''' 
    list_pred = []
    result = {}
    for rel in sgg:
        if len(rel) > 3:
            rel = rel[:3]
        for idx, node in enumerate(rel):
            if node not in list(result.keys()):
                if idx < 2:
                    result[node] = [rel[idx+1]]
                else:
                    result[node] = []
            else:
                if idx < 2:
                    result[node].append(rel[idx+1])
    if list_nodes:
        for node in list_nodes:
            if node not in list(result.keys()):
                result[node] = []
    return result

def convert_dict_to_forcegraph_format(sgg_dict):
    '''
    Convert the Flowpoint graph (output of convert_sgg_to_dict) to Forcegraph format
    Used for the Reactjs
    ''' 
    data = {}
    data['nodes'] = []
    data['links'] = []
    for node in list(sgg_dict.keys()):
        if ':' in node: # subject
            data['nodes'].append({"id": node, "group": 1})
        else: # predicate
            data['nodes'].append({"id": node, "group": 2})

    for node in list(sgg_dict.keys()):
        list_pred = sgg_dict[node]
        for pred in list_pred:
            if ':' in pred: # this pred is object
                data['links'].append({"source": node, "target": pred, "value": 1})
            else: # this pred is predicate
                data['links'].append({"source": node, "target": pred, "value": 2})
    return data

def translate_sgg_to_human_read(sgg, info_dict):
    '''
    Convert an sgg list of list into list of list but human readable
    ''' 
    result = []
    for i in range(len(sgg)):
        line_rel = sgg[i]
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