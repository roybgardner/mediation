
from IPython.display import clear_output

import networkx as nx

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles



import json
import csv


from scipy import stats
from scipy.spatial.distance import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored
import time



twenty_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',\
                          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',\
                          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',\
                          '#000000']


def load_mediation_data(mediation_file,actors_file,data_path):
    data_dict = {}
    
    # Read the main CSV
    with open(data_path + mediation_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        header = next(reader)
        # Put the remaining rows into a list of lists
        data = [row for row in reader]
    
    # Read the actors CSV
    with open(data_path + actors_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # Get the header row
        actors_header = next(reader)
        # Put the remaining rows into a list of lists
        actors_data = [row for row in reader]

    # Each data row defines a mediation-actor relationship
    # Mediation vertices
    mediation_vertices = sorted(list(set([row[header.index('mediation ID')].strip() for row in data])))
    
    # Actor vertices - using the 'third-party' column because it's used for lead actor definition
    # and in actors CSV
    actor_vertices = sorted(list(set([row[header.index('third-party')].strip() for row in data])))  
    
    # Check actor vertices from mediation against actors in actors file
    # Detected one anomaly - Netherland and The Netherlands in actors CSV but consistent use
    # of Netherlands in mediation CSV
    #actors = [row[0].strip() for row in actors_data]
    #print(len(actors))
    #print(set(actors).difference(set(actor_vertices)))
    
    # Build vertices dict
    vertices_dict = {}
    vertex_types = []
    for row in data:
        mediation_id = row[header.index('mediation ID')].strip()
        if not mediation_id in vertices_dict:
            vertices_dict[mediation_id] = {}
            vertices_dict[mediation_id]['type'] = 'MED'
            # Need some mediation data. Going to use: year, conflict location,
            # negotiation location, negotiation type, agreement
            vertices_dict[mediation_id]['data'] = {}
            year = row[header.index('year')].strip()
            vertices_dict[mediation_id]['data']['year'] = year
            conflict_locale = row[header.index('conflict locale')].strip()
            vertices_dict[mediation_id]['data']['conflict_locale'] = conflict_locale
            neg_location = row[header.index('location of neogitations')].strip()
            vertices_dict[mediation_id]['data']['neg_location'] = neg_location
            neg_type = row[header.index('negotiation type')].strip()
            vertices_dict[mediation_id]['data']['neg_type'] = neg_type        
            agreement = int(row[header.index('agreement')].strip())
            vertices_dict[mediation_id]['data']['agreement'] = agreement        
            peace_agreement = int(row[header.index('peace agreement')].strip())
            vertices_dict[mediation_id]['data']['peace_agreement'] = peace_agreement        
    for row in actors_data:
        actor_id = row[0].strip()
        actor_type = row[1].strip()
        vertices_dict[actor_id] = {}
        vertices_dict[actor_id]['type'] = actor_type
        vertices_dict[actor_id]['data'] = {}
        vertices_dict[actor_id]['data']['relative_type'] = row[2].strip()
    
    # Collect all vertex types
    vertex_types = list(set([v['type'] for k,v in vertices_dict.items()]))
    # Build a colour map for types
    color_map = {type_:twenty_distinct_colors[i] for\
                 i,type_ in enumerate(vertex_types)}
        
    # Build the biadjacency matrix
    matrix = np.zeros([len(mediation_vertices),len(actor_vertices)]).astype(np.int64)
    for row in data:
        mediation_id = row[header.index('mediation ID')]
        mediation_index = mediation_vertices.index(mediation_id)
        actor_id = row[header.index('third-party')].strip()
        actor_index = actor_vertices.index(actor_id)
        
        # Need to get the cell value
        edge_dict = populate_edge_dict(actor_id,row,header)
        
        matrix[mediation_index][actor_index] = get_edge_weight(edge_dict)
        
    data_dict['header'] = header
    data_dict['data'] = data
    data_dict['mediation_vertices'] = mediation_vertices
    data_dict['actor_vertices'] = actor_vertices
    data_dict['vertices_dict'] = vertices_dict
    data_dict['color_map'] = color_map
    data_dict['matrix'] = matrix
    return data_dict

def get_empty_edge_dict():
    """
    Dictionary for storing mediation-actor edge properties.
    The values of the properties form a binary string
    These strings are converted to integers for use a biadjacency cells values which are edge weights
    return dictionary of properties
    """
    edge_dict = {}
    edge_dict['is_lead'] = 0
    edge_dict['good_offices'] = 0
    edge_dict['mediation'] = 0
    edge_dict['hosting'] = 0
    edge_dict['negotiating'] = 0
    edge_dict['manipulating'] = 0
    edge_dict['humanitarian'] = 0
    edge_dict['witness'] = 0
    edge_dict['other'] = 0
    return edge_dict

def populate_edge_dict(actor_id,row,header):
    """
    Populate an edge dictionary from a row for a given actor
    param actor_id: ID of actor
    param row: Mediation data row
    param header: Mediation row header
    return dictionary of mediation-actor properties
    """
    edge_dict = get_empty_edge_dict()
    leads = []
    leads.append(row[header.index('leading actor')].strip())
    leads.append(row[header.index('leading actor 2')].strip())
    leads.append(row[header.index('leading actor 3')].strip())
    if actor_id in leads:
        edge_dict['is_lead'] = 1
        
    edge_dict['good_offices'] = int(row[header.index('good offices')].strip() or 0)
    edge_dict['mediation'] = int(row[header.index('mediation')].strip() or 0)
    edge_dict['hosting'] = int(row[header.index('hosting talks')].strip() or 0)
    edge_dict['negotiating'] =\
        int(row[header.index('negotiating and drafting')].strip() or 0)
    edge_dict['manipulating'] = int(row[header.index('manipulating')].strip() or 0)
    edge_dict['humanitarian'] = int(row[header.index('humanitarian')].strip() or 0)
    edge_dict['witness'] =\
        int(row[header.index('witness/party to agreement')].strip() or 0)
    edge_dict['other'] = int(row[header.index('other')].strip() or 0)
    return edge_dict

def get_edge_weight(edge_dict):
    """
    Convert edge property values into an integer
    param edge_dict: dictionary containing set of Boolean valued edge properties
    return edge_weight: an integer encoding the binary string of edge properties
    """
    s = ''
    for k,v in edge_dict.items():
        s += str(v)
    return int(''.join(c for c in s),2)

def recover_edge_dict(edge_weight,props_length):
    """
    Recover the edge dictionary from an edge weight
    param edge_weight: integer value
    param props_length: length of properties binary string
    return edge_dict: dictionary containing set of Boolean valued edge properties
    """
    formatter = '0' + str(props_length) + 'b'
    # Convert edge weight integer to binary string
    b = format(edge_weight, formatter)
    edge_dict = {}
    edge_dict['is_lead'] = b[0]
    edge_dict['good_offices'] = b[1]
    edge_dict['mediation'] = b[2]
    edge_dict['hosting'] = b[3]
    edge_dict['negotiating'] = b[4]
    edge_dict['manipulating'] = b[5]
    edge_dict['humanitarian'] = b[6]
    edge_dict['witness'] = b[7]
    edge_dict['other'] = b[8]
    return edge_dict

def adjacency_from_biadjacency(data_dict):
    """
    Build the full adjacency matrix from the binary part which represents bipartite 
    graph. The full adjacency is needed for DFS and by network packages.
    Rows and columns of the adjacency matrix are identical and
    are constructed from the biadjacency matrix in row-column order.
    The number of rows (and columns) in the adjacency matrix is therefore:
    biadjacency.shape[0] +  biadjacency.shape[1]
    param data_dict: A dictionary containing mediation network data
    return adjacency matrix and list of vertex labels. The latter is the concatenated lists of
    agreement and actor vertex labels
    """    
    binary_matrix = data_dict['matrix']
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
    adj_vertices.extend(data_dict['mediation_vertices'])
    adj_vertices.extend(data_dict['actor_vertices'])

    return adjacency_matrix,adj_vertices

def get_cooccurrence_matrices(matrix):
    bin_matrix = (matrix > 1).astype(np.int8)
    # Columns-columns co-occurence matrix
    V = np.matmul(bin_matrix.T,bin_matrix)
    # Rows-rows co-occurence matrix
    W = np.matmul(bin_matrix,bin_matrix.T)
    return (V,W)

def get_mediations(actor_id,data_dict):
    """
    Get the mediations in which an actor is engaged
    param actor_id: Actor ID
    param data_dict: Mediation data dictionary
    return: List of mediations in which the actor in actor_id is engaged
    """
    actor_index = data_dict['actor_vertices'].index(actor_id)
    mediation_ids = [(data_dict['mediation_vertices'][i],int(v),\
                      data_dict['vertices_dict'][data_dict['mediation_vertices'][i]]['data']['year']) for\
                     i,v in enumerate(data_dict['matrix'][:,actor_index]) if v > 0]
    return mediation_ids





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

def draw_networkx_graph(matrix,vertices,data_dict):
    node_labels = {i:v for i,v in enumerate(vertices)}
    node_colors = [data_dict['color_map'][data_dict['vertices_dict']\
                                          [vertex_id]['type']] for vertex_id in vertices]

    graph = nx.from_numpy_array(matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(12,12))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,\
                     node_size=400,font_size=12,alpha=0.6)
    plt.grid(False)
    plt.show()


def display_networkx_graph_ts(actor_ids,query_matrix,vertex_indices,adj_vertices,data_dict,title='',file=''):
    node_labels = {}
    node_sizes = []
    for i,index in enumerate(vertex_indices):
        node_id = adj_vertices[index]
        node_labels[i] = node_id
        
        if node_id == actor_ids[0]:
            left_node = i
            node_sizes.append(1600)
        elif node_id == actor_ids[1]:
            right_node = i
            node_sizes.append(1600)
        else:
            node_sizes.append(400)
        
    found_ids = [adj_vertices[index] for i,index in enumerate(vertex_indices)]
    node_colors = [data_dict['color_map'][data_dict['vertices_dict']\
                                          [vertex_id]['type']] for vertex_id in found_ids]
    
    graph = nx.from_numpy_array(query_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(12,12))
    pos = nx.circular_layout(graph)
    
    pos[left_node] = np.array([-2,0.1])
    pos[right_node] = np.array([2,0])

    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=node_sizes,\
                     font_size=12,alpha=0.6)
    plt.grid(False)
    plt.title(title,fontsize='xx-large')
    if len(file) > 0:
        plt.savefig('../../outputs/' + file + '.png', bbox_inches='tight')
    plt.show()

def get_joint_mediations(actor_ids,data_dict):
    """
    Given a list of actors get the mediations in common
    param actor_ids: List of actor IDs
    param data_dict: Mediations data dictionary
    return: List of mediations in which actors are engaged
    """
    # Given a list of actors get the mediations in common
    if len(actor_ids) < 2:        
        return []
    for actor_id in actor_ids:
        if not actor_id in data_dict['actor_vertices']:
            return []
    actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in actor_ids]
    for i,actor_index in enumerate(actor_indices):
        row = data_dict['matrix'].T[actor_index]
        if i == 0:
            mediation_bitset = row
        else:
            mediation_bitset = np.bitwise_and(mediation_bitset,row)
    mediation_ids = []
    for index,value in enumerate(mediation_bitset): 
        if value > 1:
            mediation_ids.append(data_dict['mediation_vertices'][index])
    return mediation_ids

    
def plot_birectional(left,right,titles,max_x=0,labelled=False,file=''):
    """
    Plot a bidirectional graph with constitution section percentages on left and responses on right.
    x-axis is percentage of above-threshold sections or responses above-threshold for a topic or category
    y-axis is topic or category label (for all topics the label is the topic ID and is not shown)
    The left and right data sets must be the same length.
    param left: Data tuples (label,value) for the left side.
    param right: Data tuples (label,value) for the right side
    param titles: Left and right titles in a list [<left_title>,<right__title>]
    param max_x: Use to manually set the max_x value. If 0 defaults to maximum x + 1 of data series values
    param labelled: If True then display data tuple lables on the y-axis, otherwise just use integer 
    values over range of data
    """
    maxs = [] 
    maxs.append(max([t[1] for t in left]))
    maxs.append(max([t[1] for t in right]))

    if max_x == 0:
        max_x = int(max(maxs))+1
    else:
        max_x = max_x

    font_color = '#525252'
    facecolor = '#eaeaf2'
    color_red = '#fd625e'
    color_blue = '#01b8aa'
    index = list(range(0,len(left)))
    column0 = [t[1] for t in left]
    column1 = [t[1] for t in right]
    title0 = titles[0]
    title1 = titles[1]

    fig, axes = plt.subplots(figsize=(10,14), ncols=2, sharey=False)
    fig.tight_layout()

    axes[0].barh(index, column0, align='center', color=color_red, zorder=10)
    axes[0].set_title(title0, fontsize=16, pad=15, color=color_red)
    axes[1].barh(index, column1, align='center', color=color_blue, zorder=10)
    axes[1].set_title(title1, fontsize=16, pad=15, color=color_blue)

    axes[0].invert_xaxis() 
    #plt.gca().invert_yaxis()

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)

    axes[0].set_xticks(list(range(0,max_x+5,10)))
    axes[1].set_xticks(list(range(0,max_x+5,10)))
    axes[1].set_yticks([])

    if labelled:
        axes[0].set_yticks(range(0,len(left)))
        axes[0].set_yticklabels([t[0] for t in left])
    else:
        axes[0].set_ylabel('Topic index',fontsize='x-large',)
        

    axes[0].tick_params(labelsize='x-large')
    axes[1].tick_params(labelsize='x-large')
    #axes[0].set_ylabel('Ranked topics',fontsize='xx-large')
    axes[0].set_xlabel('Number of lead actor roles',fontsize='x-large',)
    axes[1].set_xlabel('Number of lead actor roles',fontsize='x-large',)

    #axes[0].xaxis.set_label_coords(1.025, -0.055)
    if len(file) > 0:
        plt.savefig('./figures/' + file + '.tif',dpi=300,bbox_inches='tight')
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
    node_colors = [data_dict['color_map'][data_dict['vertices_dict']\
                                          [vertex_id]['type']] for vertex_id in vertex_list]
    graph = nx.from_numpy_array(co_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(12,12))
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
