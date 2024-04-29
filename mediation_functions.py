import networkx as nx

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import json
import csv


from scipy import stats
from scipy.spatial.distance import *


twenty_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',\
                          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',\
                          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',\
                          '#000000']

def depth_first_search(matrix,query_index,max_depth=1,depth=1,vertices=[],visited=[]):
    """
    Recursive function to visit all vertices that are reachable from a query vertex.
    param matrix: The adjacency matrix representation of a graph
    param query_index: The row/column index which defines the query vertex
    param max_depth: How deep to go (for a bipartite graph the maximum is 2)
    param depth: Keeps track of how deep we have gone
    param vertices: Store found vertices
    param visited: Store visited vertices so we have a terminating condition
    return list of vertices
    """
    visited.append(query_index)
    # Index row - find connected head vertices in the query index row. In other words,
    # find the vertices that the query vertex point to
    vertices.extend([i for i,v in enumerate(matrix[query_index]) if v > 0 and not i in visited])
    if depth < max_depth:
        for i in vertices:
            if i in visited:
                continue
            vertices = depth_first_search(matrix,i,max_depth=1,depth=1,vertices=vertices,visited=visited)
    return vertices

def adjacency_from_biadjacency(pp_data_dict):
    """
    Build the full adjacency matrix from the binary part which represents bipartite 
    agreements-actor graph. The full adjacency is needed for DFS and by network packages.
    Rows and columns of the adjacency matrix are identical and
    are constructed from the binary-valued matrix in row-column order.
    The number of rows (and columns) in the adjacency matrix is therefore:
    binary_matix.shape[0] +  binary_matix.shape[1]
    param pp_data_dict: A dictionary containing peace process network data inclusing the BVRM
    return adjacency matrix and list of vertex labels. The latter is the concatenated lists of
    agreement and actor vertex labels
    """    
    binary_matrix = pp_data_dict['pp_matrix']
    size = binary_matrix.shape[0] + binary_matrix.shape[1]
    adjacency_matrix = np.zeros((size,size))
    
    # Get the range of the bin matrix rows to generate the upper triangle
    # of the adjacency matrix
    row_index = 0
    col_index = binary_matrix.shape[0]
    adjacency_matrix[row_index:row_index + binary_matrix.shape[0],\
           col_index:col_index + binary_matrix.shape[1]] = binary_matrix
    # Add in the lower triangle
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T    
    adj_vertices = []
    adj_vertices.extend(pp_data_dict['pp_agreement_ids'])
    adj_vertices.extend(pp_data_dict['pp_actor_ids'])

    return adjacency_matrix,adj_vertices

def get_query_matrix(query_indices,matrix,max_depth=1,operator='OR'):    
    """
    Query an adjacency matrix using depth-first search
    param query_indices: The indices of the query vertices
    param matrix: The adjacency matrix we are querying
    param max_depth: Max depth of the search. Defaults to 1. Agreement-actor graphs are bipartite so
    the maximum depth is 2.
    param operator: Boolean operator to use on found vertices. AND restricts the results to entities
    that have an edge to all query vertices.
    return: An adjacency matrix for the set of found vertices and the indices of the found vertices
    """    
    found_indices = []
    for i,query_index in enumerate(query_indices):
        vertices = depth_first_search(matrix,query_index,max_depth=max_depth,vertices=[],visited=[])
        if i == 0:
            found_indices.extend(vertices)
        else:
            if operator == 'OR':
                found_indices = list(set(found_indices).union(set(vertices)))
            else:
                found_indices = list(set(found_indices).intersection(set(vertices)))
    # Add the query vertex to the found vertices
    found_indices.extend(query_indices)    
    found_indices = sorted(found_indices)
    # Extract the sub-matrix containing only the found vertices
    query_matrix = matrix[np.ix_(found_indices,found_indices)]
    return query_matrix,found_indices

def display_networkx_graph(query_matrix,vertex_indices,adj_vertices,data_dict):
    node_labels = {i:adj_vertices[index] for i,index in enumerate(vertex_indices)}
    node_colors = [data_dict['color_map'][v.split('_')[0]] for _,v in node_labels.items()]
    graph = nx.from_numpy_array(query_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=400,font_size=12,alpha=0.6)
    plt.grid(False)
    plt.show()
    
def display_comatrix_as_networkx_graph(co_matrix,vertex_indices,vertex_list,data_dict,title=''):
    """
    Create and display a networkx graph of a co-occurrence matrix. Includes the display of singletons — vertices that have no co-occurrences.
    param co_matrix: Co-occurence matrix (only the uppoer triangle is used to avoid self loops and edge duplication)
    param vertex_indices: The indices of the vertices in the occurrence matrix. These are indices into the complete set of vertices of which the
    co-occurrence vertices may be a subset
    param vertex_list: Complete list of vertex identifiers. The indices in vertex_indices locate the identifiers for the co-occurring vertices.
    param data_dict: The application data dictionary.
    param title: Optional title.
    """
    co_matrix = np.triu(co_matrix,k=1)
    node_labels = {i:vertex_list[index] for i,index in enumerate(vertex_indices)}
    node_colors = [data_dict['color_map'][v.split('_')[0]] for _,v in node_labels.items()]
    graph = nx.from_numpy_array(co_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(16,16))
    pos = nx.circular_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=400,alpha=0.6)
    # Get the edge labels
    rc = np.nonzero(co_matrix) # Row and column indices of non-zero pairs
    z = list(zip(list(rc[0]),list(rc[1])))
    edge_labels = {t:co_matrix[t[0]][t[1]] for t in z}
    nx.draw_networkx_edge_labels(graph, pos,edge_labels,font_color='red',font_size=12)
    plt.grid(False)
    plt.title(title)
    plt.show()

def get_peace_processes(data_dict):
    """
    Get list of peace process names 
    param data_dict: The application's data dictionary obtained from load_agreement_actor_data()
    return: list of process names in alpha order
    """
    processes = [row[data_dict['links_header'].index('PPName')].strip() for row in data_dict['links_data']]
    return sorted(list(set(processes)))

def get_peace_process_data(process_name,data_dict):
    
    # Peace process data are in the links table so collect all edges assigned to the process
    pp_edges = [row for row in data_dict['links_data'] if row[data_dict['links_header'].\
                                                              index('PPName')].strip()==process_name]
    
    # Now we want the indices of peace process agreements and actors so we can extract the peace process
    # sub-matrix
    pp_agreement_ids = list(set([row[data_dict['links_header'].index('from_node_id')] for row in pp_edges]))
    pp_agreement_indices = [data_dict['agreement_vertices'].index(agreement_id) for\
                            agreement_id in pp_agreement_ids]
    
    pp_actor_ids = list(set([row[data_dict['links_header'].index('to_node_id')] for row in pp_edges]))
    pp_actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in pp_actor_ids]

    pp_matrix = data_dict['matrix'][np.ix_(pp_agreement_indices,pp_actor_indices)]
    pp_matrix = np.array(pp_matrix)
    pp_data_dict = {}
    pp_data_dict['pp_actor_ids'] = pp_actor_ids
    pp_data_dict['pp_agreement_ids'] = pp_agreement_ids
    pp_data_dict['pp_matrix'] = pp_matrix    
    return pp_data_dict

def get_cooccurrence_matrices(matrix):
    # Actor-actor co-occurence matrix for a peace process
    V = np.matmul(matrix.T,matrix)
    # Agreement-agreement co-occurence matrix
    W = np.matmul(matrix,matrix.T)
    return (V,W)

def load_agreement_actor_data(nodes_file,links_file,agreements_dict,data_path):
    # Stash data in a dictionary
    data_dict = {}
    
    # Read the CSVs
    with open(data_path + nodes_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        nodes_header = next(reader)
        # Put the remaining rows into a list of lists
        nodes_data = [row for row in reader]

    with open(data_path + links_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        links_header = next(reader)
        # Put the remaining rows into a list of lists
        links_data = [row for row in reader]

    with open(data_path + agreements_dict) as f:
        agreements_dict = json.load(f)
    
    # Agreement are from vertices
    agreement_vertices = list(set([row[links_header.index('from_node_id')] for row in links_data]))
    # Actors are to vertices
    actor_vertices = list(set([row[links_header.index('to_node_id')] for row in links_data]))

    # edge_dict links each agreement to a list of actors
    edge_dict = {}
    # dates_dict stores link each agreement to an integer date in YYYYMMDD format
    dates_dict = {}
    for row in links_data:
        if row[links_header.index('from_node_id')] in edge_dict:
            edge_dict[row[links_header.index('from_node_id')]].append(row[links_header.index('to_node_id')])
        else:
            edge_dict[row[links_header.index('from_node_id')]] = [row[links_header.index('to_node_id')]]
        if not row[5] in dates_dict:
            a = row[1].split('-')
            dates_dict[row[links_header.index('from_node_id')]] = int(''.join(a))
    
    # Build a vertices dictionary with node_id as key and node row as the value
    vertices_dict = {row[nodes_header.index('node_id')]:row for row in nodes_data}

    # Collect all vertex types
    vertex_types = []
    for k,v in vertices_dict.items():
        type_ = v[nodes_header.index('type')]
        if len(type_) == 0:
            # This type is missing in node data
            type_ = 'AGT'
        vertex_types.append(type_)
    vertex_types = sorted(list(set(vertex_types)))

    # Build a colour map for types
    color_map = {type_:twenty_distinct_colors[i] for\
                 i,type_ in enumerate(vertex_types)}
        
    # Build the agreement-actor biadjacency matrix - the core data structure
    matrix = []
    for agreement in agreement_vertices:
        row = [0]*len(actor_vertices)
        for i,actor in enumerate(actor_vertices):
            if actor in edge_dict[agreement]:
                row[i] = 1
        matrix.append(row)
    matrix = np.array(matrix)
    
    data_dict['agreements_dict'] = agreements_dict
    data_dict['dates_dict'] = dates_dict
    data_dict['nodes_data'] = nodes_data
    data_dict['nodes_header'] = nodes_header
    data_dict['links_data'] = links_data
    data_dict['links_header'] = links_header
    data_dict['agreement_vertices'] = agreement_vertices
    data_dict['actor_vertices'] = actor_vertices
    data_dict['vertices_dict'] = vertices_dict
    data_dict['color_map'] = color_map
    data_dict['matrix'] = matrix

    return data_dict

def get_agreement_cosignatories(agreement_ids,pp_data_dict):
    """
    Given a list of agreements get the signatories in common
    Works within a peace process only
    param agreement_ids: List of agreement IDs
    param pp_data_dict: Peace process data dictionary
    return: List of actor IDs who a signatories to all the agreements in agreement_ids
    """
    if len(agreement_ids) < 2:        
        return []
    for agreement_id in agreement_ids:
        if not agreement_id in pp_data_dict['pp_agreement_ids']:
            return []
    agreement_indices = [pp_data_dict['pp_agreement_ids'].index(agreement_id) for\
                         agreement_id in agreement_ids]
    for i,agreement_index in enumerate(agreement_indices):
        row = pp_data_dict['pp_matrix'][agreement_index]
        if i == 0:
            actors_bitset = row
        else:
            actors_bitset = np.bitwise_and(actors_bitset,row)
    actor_ids = []
    for index,value in enumerate(actors_bitset): 
        if value == 1:
            actor_ids.append(pp_data_dict['pp_actor_ids'][index])
    return actor_ids

def get_consignatory_agreements_from_data_dict(actor_ids,data_dict):
    """
    Given a list of actors get the agreements in common form the entire data set
    param actor_ids: List of actor IDs
    param data_dict: Data dictionary
    return: List of agreements to which the actors in actor_ids are cosignatories
    """
    # Given a list of actors get the agreements in common
    if len(actor_ids) < 2:        
        return []
    for actor_id in actor_ids:
        if not actor_id in data_dict['actor_vertices']:
            return []
    actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in actor_ids]
    for i,actor_index in enumerate(actor_indices):
        row = data_dict['matrix'].T[actor_index]
        if i == 0:
            agreements_bitset = row
        else:
            agreements_bitset = np.bitwise_and(agreements_bitset,row)
    agreement_ids = []
    for index,value in enumerate(agreements_bitset): 
        if value == 1:
            agreement_ids.append(data_dict['agreement_vertices'][index])
    return agreement_ids


def get_consignatory_agreements(actor_ids,pp_data_dict):
    """
    Given a list of actors get the agreements in common
    Works within a peace process only
    param actor_ids: List of actor IDs
    param pp_data_dict: Peace process data dictionary
    return: List of agreements to which the actors in actor_ids are cosignatories
    """
    # Given a list of actors get the agreements in common
    if len(actor_ids) < 2:        
        return []
    for actor_id in actor_ids:
        if not actor_id in pp_data_dict['pp_actor_ids']:
            return []
    actor_indices = [pp_data_dict['pp_actor_ids'].index(actor_id) for actor_id in actor_ids]
    for i,actor_index in enumerate(actor_indices):
        row = pp_data_dict['pp_matrix'].T[actor_index]
        if i == 0:
            agreements_bitset = row
        else:
            agreements_bitset = np.bitwise_and(agreements_bitset,row)
    agreement_ids = []
    for index,value in enumerate(agreements_bitset): 
        if value == 1:
            agreement_ids.append(pp_data_dict['pp_agreement_ids'][index])
    return agreement_ids

def get_consignatories(actor_id,pp_data_dict):
    """
    Get the cosignatories of an actor
    Works within a peace process only
    param actor_id: Actor ID
    param pp_data_dict: Peace process data dictionary
    return: List of actors who are cosignatories with the actor in actor_id
    """
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    actor_index = pp_data_dict['pp_actor_ids'].index(actor_id)
    cosign_ids = [pp_data_dict['pp_actor_ids'][i] for i,v in enumerate(co_matrices[0][actor_index]) if v > 0]
    return cosign_ids

def get_coagreements(agreement_id,pp_data_dict):
    """
    Get the coagreements of an agreement, i.e., the agreements that have signatories in 
    common with the agreement in agreement_id
    Works within a peace process only
    param agreement_id: agreement ID
    param pp_data_dict: Peace process data dictionary
    return: List of agreements with actors in common with the agreement in agreement_id
    """
    co_matrices = get_cooccurrence_matrices(pp_data_dict['pp_matrix'])
    agreement_index = pp_data_dict['pp_agreement_ids'].index(agreement_id)
    coagree_ids = [pp_data_dict['pp_agreement_ids'][i] for\
                   i,v in enumerate(co_matrices[1][agreement_index]) if v > 0]
    return coagree_ids

def get_agreements(actor_id,pp_data_dict):
    """
    Get the agreements to which an actor is a signatory
    Works within a peace process only
    param actor_id: Actor ID
    param pp_data_dict: Peace process data dictionary
    return: List of agreements to which the actor in actor_id is a signatory
    """
    actor_index = pp_data_dict['pp_actor_ids'].index(actor_id)
    agreement_ids = [pp_data_dict['pp_agreement_ids'][i] for\
                     i,v in enumerate(pp_data_dict['pp_matrix'][:,actor_index]) if v > 0]
    return agreement_ids

def get_actors(agreement_id,pp_data_dict):
    """
    Get the actors who are signatories to the agreement in agreement_id
    Works within a peace process only
    param agreement_id: agreement ID
    param pp_data_dict: Peace process data dictionary
    return: List of actors who a signatories to the agreement in agreement_id
    """
    agreement_index = pp_data_dict['pp_agreement_ids'].index(agreement_id)
    actor_ids = [pp_data_dict['pp_actor_ids'][i] for\
                     i,v in enumerate(pp_data_dict['pp_matrix'][agreement_index]) if v > 0]
    return actor_ids

def get_actor_name(actor_id,data_dict):
    """
    Get the name of an actor
    param actor_id: actor ID
    param data_dict: Global data dictionary
    return: Name of actor
    """
    return data_dict['vertices_dict'][actor_id][data_dict['nodes_header'].index('node_name')]

def get_agreement_name(agreement_id,data_dict):
    """
    Get the name of an agreement
    param agreement_id: agreement ID
    param data_dict: Global data dictionary
    return: Name of agreement
    """
    return data_dict['vertices_dict'][agreement_id][data_dict['nodes_header'].index('node_name')]

def get_agreement_date(agreement_id,data_dict):
    """
    Get the date of an agreement
    param agreement_id: agreement ID
    param data_dict: Global data dictionary
    return: Name of agreement
    """
    return data_dict['vertices_dict'][agreement_id][data_dict['nodes_header'].index('date')]
