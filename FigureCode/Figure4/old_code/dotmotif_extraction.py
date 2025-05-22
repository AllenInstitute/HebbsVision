import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
import pandas
import json
from tqdm import tqdm
import pickle
 
def calc_ground_truth(pre_cells, synapse, post_cells, use_sizes=False):
    total_synapse_count = 0
    pre_cell_pt_index = pre_cells['pt_root_id']
    post_cell_pt_index = post_cells['pt_root_id']
    synapses = synapse[(synapse['pre_pt_root_id'].isin(pre_cells['pt_root_id'])) & (synapse['post_pt_root_id'].isin(post_cells['pt_root_id']))]
    pre_cell_pt_index = pre_cell_pt_index.to_list()
    post_cell_pt_index = post_cell_pt_index.to_list()
 
    synapse_connectome = np.zeros((len(pre_cells), len(post_cells)))
    for synapse in tqdm(range(len(synapses))):
        preidx = pre_cell_pt_index.index(synapses.iloc[synapse]['pre_pt_root_id'])
        postidx = post_cell_pt_index.index(synapses.iloc[synapse]['post_pt_root_id'])
        if use_sizes:
            synapse_connectome[preidx,postidx] += synapses.iloc[synapse]['size']
        else:
            synapse_connectome[preidx,postidx] += 1
        total_synapse_count += 1
    print("Total Synpase Count in Weight Matrix:", total_synapse_count)
    return synapse_connectome

# Import Stefan's Library for Data Management of V1DD
from LSMMData.data_management import LSMMData

with open('v1dd_wrangler_input_dict.json') as f:
    loaded_json = json.load(f)
my_data = LSMMData.LSMMData(loaded_json)
tables = my_data.data
params = my_data.params
dirs = my_data.dirs
mappings = my_data.mappings

ground_truth_connectome_v1dd = calc_ground_truth(tables['structural']['pre_cell'], tables['structural']['synapse'], tables['structural']['post_cell'])
adjacency_matrix = ground_truth_connectome_v1dd.clip(0,1)
# Ensure that the diagonal is filled with zeros
np.fill_diagonal(adjacency_matrix, 0)
pyr_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Extracts Three Chain Motifs
executor = GrandIsoExecutor(graph=pyr_graph)
three_chain = Motif("""
                A -> B
                B -> C
              """)
three_chain_results = executor.find(three_chain)
print('\n\t',len(three_chain_results))
with open('dot_motif_results/pyc_three_chain_results.pickle', 'wb') as out_file:
    pickle.dump(three_chain_results, out_file)

# # Extracts Four Chain Motifs
# four_chain = Motif("""
#                 A -> B
#                 B -> C
#                 C -> D
#               """)

# four_chain_results = executor.find(four_chain)
# print('\n\t',len(four_chain_results))
# with open('dot_motif_results/pyc_four_chain_results.pickle', 'wb') as out_file:
#     pickle.dump(four_chain_results, out_file)