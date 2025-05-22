import numpy as np
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
import pandas
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import statsmodels.stats.multitest as smm
import v1dd_coregistration_wrangling
 
def calc_ground_truth(pre_cells, synapse, post_cells, use_sizes=False):
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
    return synapse_connectome

# Full Matrix
# weight_matrix = np.load('ground_truth_summed_weights_v1dd_668.npy')
# adjacency_matrix = weight_matrix.clip(0,1)
# graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

tables, mappings = v1dd_coregistration_wrangling.get_tables_and_mappings(online=True)

# Make a graph of just excitatory cells
cell_table = tables['cell']

# # cell_table = cell_table.query('soma_layer == "23"')[['pt_root_id', 'soma_layer']].reset_index()
cell_table['connectome_index'] = cell_table.index

# cell_table['connectome_index'] = cell_table.index
# cell_table = cell_table.query('cell_type == "PYC"')[['connectome_index', 'pt_root_id', 'soma_layer']].reset_index()

synapse_table = pandas.read_pickle('synapse_Table_668.pickle')
weight_matrix = calc_ground_truth(cell_table, synapse_table, cell_table, use_sizes=True)
adjacency_matrix = weight_matrix.clip(0,1)
pyr_graph = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

# Motif Analysis with DotMotif
executor = GrandIsoExecutor(graph=pyr_graph)

three_chain = Motif("""
                A -> B
                B -> C
              """)

three_chain_results = executor.find(three_chain)

print('\n\t',len(three_chain_results))
with open('pyc_stefan_motif_three_chain_results.pickle', 'wb') as out_file:
    pickle.dump(three_chain_results, out_file)

# length four chains, to do after three

# four_chain = Motif("""
#                 A -> B
#                 B -> C
#                 C -> D
#               """)

# four_chain_results = executor.find(four_chain)

# # print(four_chain_results)
# with open('pyc_stefan_motif_four_chain_results.pickle', 'wb') as out_file:
#     pickle.dump(four_chain_results, out_file)

# Now we sort by assembly and test
    
def invert_dict(original_dict):
    inverted_dict = {}
    for key, items in original_dict.items():
        for item in items:
            if item not in inverted_dict:
                inverted_dict[item] = [key]
            else:
                inverted_dict[item].append(key)
    return inverted_dict

def update_dict_keys(key_mapping_dict, dict_to_update):
    return {key_mapping_dict.get(key, key): val for key, val in dict_to_update.items()}

# Load assembly and coregistration data, and create useful mappings
functional_cell_indexes_by_assembly = tables['functional_indexes_by_assembly']
assemblies_by_functional_index = tables['assemblies_by_functional_index']
coregistered_cells = tables['coregistration']
connectome_id_to_root_id_mapping = mappings['connectome_id_to_root_id']
# connectome_id_to_functional_id_mapping = dict(coregistered_cells[['unit_id', 'id']].values)
# functional_id_to_pt_root_id_mapping = dict(coregistered_cells[['id', 'pt_root_id']].values)
# connectome_id_to_root_id_mapping = dict(cell_table[['connectome_index', 'pt_root_id']].values)
assemblies_by_connectome_index = tables['assemblies_by_connectome_id'] #update_dict_keys(connectome_id_to_functional_id_mapping, assemblies_by_functional_index)
coregistered_unit_ids = list(coregistered_cells['unit_id'].values)

# Set up storage for assembly sorting
psd_size_list_by_assembly = {}
interior_chains_by_assembly = {}
chains_between_assemblies = {}
interior_chains_pooled_assemblies = 0
interior_chains_no_assembly = 0
chain_membership_counts_by_assembly_by_cell = {}
chain_membership_counts_by_assembly_by_cell['No Assembly'] = {}

for assembly_id in list(functional_cell_indexes_by_assembly.keys()):
    psd_size_list_by_assembly[f'Assembly {assembly_id+1}'] = [] # Initialize the dict with an empty list for each assembly
    interior_chains_by_assembly[f'Assembly {assembly_id+1}'] = 0
    chains_between_assemblies[f'Assembly {assembly_id+1}'] = {}
    chain_membership_counts_by_assembly_by_cell[f'Assembly {assembly_id+1}'] = {}
    for second_assembly_id in list(functional_cell_indexes_by_assembly.keys()):
        chains_between_assemblies[f'Assembly {assembly_id+1}'][f'Assembly {second_assembly_id+1}'] = 0

psd_sizes_pooled_assemblies = []
psd_sizes_not_in_assemblies = []
psd_sizes_crossing_into_assemblies = []
psd_sizes_crossing_outof_assemblies = []

nonconnected_motifs = 0
motifs_with_origin_and_terminal_coregistered = 0
motifs_without_origin_and_terminal_coregistered = 0

for chain in tqdm(three_chain_results):
    origin_cell_in_pyc_id, middle_cell_in_pyc_id, terminal_cell_in_pyc_id = chain['A'], chain['B'], chain['C']

    # Get the connectome (out of all cells, not just PYC) and root IDs for origin, middle, and terminal chain cells
    origin_cell_in_connectome_id = cell_table.iloc[origin_cell_in_pyc_id].connectome_index
    origin_cell_in_root_id = cell_table.iloc[origin_cell_in_pyc_id].pt_root_id

    middle_cell_in_connectome_id = cell_table.iloc[middle_cell_in_pyc_id].connectome_index
    middle_cell_in_root_id = cell_table.iloc[middle_cell_in_pyc_id].pt_root_id

    terminal_cell_in_connectome_id = cell_table.iloc[terminal_cell_in_pyc_id].connectome_index
    terminal_cell_in_root_id = cell_table.iloc[terminal_cell_in_pyc_id].pt_root_id

    origin_assembly_ids = None
    terminal_assembly_ids = None
    
    if (origin_cell_in_connectome_id in coregistered_unit_ids) and (terminal_cell_in_connectome_id in coregistered_unit_ids): # If both are coregistered
        motifs_with_origin_and_terminal_coregistered += 1

        if origin_cell_in_connectome_id in tables['assemblies_by_connectome_id'].keys():
            origin_assembly_ids = list(tables['assemblies_by_connectome_id'][origin_cell_in_connectome_id])
        if terminal_cell_in_connectome_id in tables['assemblies_by_connectome_id'].keys():
            terminal_assembly_ids = list(tables['assemblies_by_connectome_id'][terminal_cell_in_connectome_id])

        # We now have origin and terminal assembly IDs.

        psd_sizes_origin_to_mid = list(synapse_table.query('pre_pt_root_id == @origin_cell_in_root_id and post_pt_root_id == @middle_cell_in_root_id')['size'].values)
        psd_sizes_mid_to_terminal = list(synapse_table.query('pre_pt_root_id == @middle_cell_in_root_id and post_pt_root_id == @terminal_cell_in_root_id')['size'].values)

        if len(psd_sizes_origin_to_mid) == 0 or len(psd_sizes_mid_to_terminal) == 0: # If we for some reason don't have chains that connect to each other?!
            nonconnected_motifs += 1
            print("\n\tHUGE ERROR!  CHAIN CELLS AREN'T CONNECTED!!!!\n")
        else:
            # Do all combinations of assembly IDs, just for this first option.
            if origin_assembly_ids is not None and terminal_assembly_ids is not None:  # If we can identify assemblies for both
                # Add to pooled assemblies
                for size in psd_sizes_origin_to_mid:
                    psd_sizes_pooled_assemblies.append(size)
                for size in psd_sizes_mid_to_terminal:
                    psd_sizes_pooled_assemblies.append(size) 
                interior_chains_pooled_assemblies += 1
                # Then add to each individual matched combo between origin and terminal assemblies
                for temp_origin_assembly_id in origin_assembly_ids:
                    for temp_terminal_assembly_id in terminal_assembly_ids:
                        if temp_origin_assembly_id == temp_terminal_assembly_id: # If they're both in the same assembly
                            # Add to specific assembly
                            for size in psd_sizes_origin_to_mid:
                                psd_size_list_by_assembly[f'Assembly {temp_origin_assembly_id+1}'].append(size)
                            for size in psd_sizes_mid_to_terminal:
                                psd_size_list_by_assembly[f'Assembly {temp_origin_assembly_id+1}'].append(size)
                            interior_chains_by_assembly[f'Assembly {temp_origin_assembly_id+1}'] += 1
                            # And add to the count of chains each cell participates in.
                            if origin_cell_in_root_id in chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'].keys():
                                chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'][origin_cell_in_root_id] += 1
                            else:
                                chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'][origin_cell_in_root_id] = 1
                            if terminal_cell_in_root_id in chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'].keys():
                                chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'][terminal_cell_in_root_id] += 1
                            else:
                                chain_membership_counts_by_assembly_by_cell[f'Assembly {temp_origin_assembly_id+1}'][terminal_cell_in_root_id] = 1
                            

                        else: # If they're not the same
                            chains_between_assemblies[f'Assembly {temp_origin_assembly_id+1}'][f'Assembly {temp_terminal_assembly_id+1}'] += 1

                
            # For the rest, we rely just on the first ones.  If something belongs to multiple assemblies, it doesn't matter for these pools.
            if origin_assembly_ids is not None and terminal_assembly_ids is not None: # If neither is in any assembly
                interior_chains_no_assembly += 1
                # And add to the count of chains each cell non-assembly origin and target cell participates in.
                if origin_cell_in_root_id in chain_membership_counts_by_assembly_by_cell[f'No Assembly'].keys():
                    chain_membership_counts_by_assembly_by_cell[f'No Assembly'][origin_cell_in_root_id] += 1
                else:
                    chain_membership_counts_by_assembly_by_cell[f'No Assembly'][origin_cell_in_root_id] = 1
                if terminal_cell_in_root_id in chain_membership_counts_by_assembly_by_cell[f'No Assembly'].keys():
                    chain_membership_counts_by_assembly_by_cell[f'No Assembly'][terminal_cell_in_root_id] += 1
                else:
                    chain_membership_counts_by_assembly_by_cell[f'No Assembly'][terminal_cell_in_root_id] = 1
                # if cell_table.iloc[origin_cell_in_pyc_id].soma_layer in ['23', '4']: # If this is a chain from 2/3/4 to 2/3/4, to allow chains to move through other layers, but compare ones that start and end among the same cell types as those in our functional assemblies from Session 1 Scan 3
                #     if cell_table.iloc[terminal_cell_in_pyc_id].soma_layer in ['23', '4']: # Wait, coregistration should take care of this automatically.  Only coregistered cells.
                for size in psd_sizes_origin_to_mid:
                    psd_sizes_not_in_assemblies.append(size)
                for size in psd_sizes_mid_to_terminal:
                    psd_sizes_not_in_assemblies.append(size)
            if origin_assembly_ids is not None and terminal_assembly_ids is not None: # If it crosses into an assembly
                # if cell_table.iloc[origin_cell_in_pyc_id].soma_layer in ['23', '4']: # If this is a chain from 2/3/4
                for size in psd_sizes_origin_to_mid:
                    psd_sizes_crossing_into_assemblies.append(size)
                for size in psd_sizes_mid_to_terminal:
                    psd_sizes_crossing_into_assemblies.append(size)
            if origin_assembly_ids is not None and terminal_assembly_ids is not None: # If it crosses out of an assembly
                # if cell_table.iloc[terminal_cell_in_pyc_id].soma_layer in ['23', '4']: # If it goes to 2/3/4
                for size in psd_sizes_origin_to_mid:
                    psd_sizes_crossing_outof_assemblies.append(size)
                for size in psd_sizes_mid_to_terminal:
                    psd_sizes_crossing_outof_assemblies.append(size)
    else:
        motifs_without_origin_and_terminal_coregistered += 1

print("Nonconnected Chains: ", nonconnected_motifs)

interior_assembly_rates = []
# interior_assembly_rates.append(interior_chains_no_assembly / )
for assembly_id in list(interior_chains_by_assembly.keys()):
    interior_assembly_rates.append(interior_chains_by_assembly[assembly_id] / len(functional_cell_indexes_by_assembly[int(assembly_id[-1])]))

print(interior_assembly_rates)

print("Motifs with start and end coregistered", motifs_with_origin_and_terminal_coregistered)
print("Motifs with start and end not coregistered (excluded from analysis)", motifs_without_origin_and_terminal_coregistered)
print("Motifs by assembly (start and end cells in the same assembly)", interior_chains_by_assembly)
print("Motifs Between Assembly Pairs:", chains_between_assemblies)
print("Pooled Assembly Interior Chains:", interior_chains_pooled_assemblies)
print("Chains to/from known nonassembly cells", interior_chains_no_assembly)

non_assembly_participation_count_list = []
for cell_id in chain_membership_counts_by_assembly_by_cell['No Assembly'].keys():
    non_assembly_participation_count_list.append(chain_membership_counts_by_assembly_by_cell['No Assembly'][cell_id]) 
print(f'No Assembly')
print(f'\tMean Participation: {np.mean(non_assembly_participation_count_list)}')

for assembly_id in list(functional_cell_indexes_by_assembly.keys()):
    participation_count_list = []
    # for cell_id in chain_membership_counts_by_assembly_by_cell[f'Assembly {assembly_id+1}'].keys():
    #     participation_count_list.append(chain_membership_counts_by_assembly_by_cell[f'Assembly {assembly_id+1}'][cell_id]) 
    print(f'Assembly {assembly_id+1}')
    participation = interior_chains_by_assembly
    print(f'\tMean Participation: {np.mean(participation_count_list)}')
    print("Mean Difference (effect size estimate):", (np.mean(participation_count_list) - np.mean(non_assembly_participation_count_list)))
    print("Standard Deviations: \n\tin-assembly:", np.std(participation_count_list), "\n\tnot-in-assembly:", np.std(non_assembly_participation_count_list))
    stat, p_value = ranksums(participation_count_list, non_assembly_participation_count_list, 'greater')
    print()

    plt.figure()
    plt.title(f'Assembly {assembly_id+1} Chain Membership Counts\np={p_value:.4f}')
    plt.hist(participation_count_list, color='goldenrod') #, bins=range(0, 1700, 50)
    plt.hist(non_assembly_participation_count_list,color='darkorchid') #, bins=range(0, 1700, 50)
    plt.legend([f'Assembly {assembly_id+1} Cells', 'Cells Not in Assemblies'])
    plt.savefig(f'Chain Length Three Participation Assembly {assembly_id+1}.png')


plt.legend(legend_list)
plt.title('Third-Order Structural PSD sizes by Functional Assembly Groups')
plt.savefig('pyc_34chain_stefan_third_order_PSD_sizes_by_assembly.png')

print('p-values', p_values)
adjusted_p_values = smm.fdrcorrection(p_values, alpha=0.05, method='indep')[1]
print('FDR p-values', adjusted_p_values)
    
# # Save work at this point
# with open('pyc_668_stefan_psd_sizes_by_assembly_dict.pickle', 'wb') as out_file:
#     pickle.dump(psd_size_list_by_assembly, out_file)

# np.save('pyc_668_stefan_psd_sizes_pooled_assemblies.npy', psd_sizes_pooled_assemblies)
# np.save('pyc_668_stefan_psd_sizes_not_in_assemblies.npy', psd_sizes_not_in_assemblies)
# np.save('pyc_668_stefan_psd_sizes_crossing_into_assemblies.npy', psd_sizes_crossing_into_assemblies)
# np.save('pyc_668_stefan_psd_sizes_crossing_outof_assemblies.npy', psd_sizes_crossing_outof_assemblies)

# # Plot and generate p-values
# p_values = [] 

# plt.figure(figsize=(30,10))
# legend_list = []

# non_assembly_sizes = np.squeeze(np.array(psd_sizes_not_in_assemblies)).flatten()
# plt.scatter([int(-20) for i in range(len(non_assembly_sizes))], list(non_assembly_sizes), c='k')
# print(f'Non-Assembly Mean Size:', np.mean(non_assembly_sizes))
# legend_list.append(f'Non-Assembly')
# sizes_array = np.squeeze(np.array(psd_sizes_pooled_assemblies)).flatten()
# plt.scatter([int(-10) for i in range(len(sizes_array))], list(sizes_array))
# print(f'Pooled Assemblies Mean Size:', np.mean(sizes_array))
# _, p_value = ranksums(non_assembly_sizes, sizes_array, alternative='less')
# print('Test: ', p_value)
# legend_list.append(f'Pooled Assemblies')

# for assembly in range(len(psd_size_list_by_assembly.keys())):
#     if len(psd_size_list_by_assembly[f'Assembly {assembly+1}']) > 0:
#         sizes_array = np.squeeze(np.array(psd_size_list_by_assembly[f'Assembly {assembly+1}'])).flatten()
#         # sizes_array = sizes_array
#         # print(sizes_array)
#         # bin_boundaries = np.arange(0,np.max(sizes_array)+np.max(sizes_array)/100,np.max(sizes_array)/100)
#         # print(len(sizes_by_assembly_dict[f'Assembly {assembly+1}']))
#         # plt.hist(sizes_by_assembly_dict[f'Assembly {assembly+1}'], bins=bin_boundaries, density=True)
#         plt.scatter([int(assembly) for i in range(len(sizes_array))], list(sizes_array))
#         print(f'Assembly {assembly+1} Mean Size:', np.mean(sizes_array))
#         _, p_value = ranksums(non_assembly_sizes, sizes_array, alternative='less')
#         p_values.append(p_value)
#         legend_list.append(f'Assembly {assembly+1}')



# plt.legend(legend_list)
# plt.title('Third-Order Structural PSD sizes by Functional Assembly Groups')
# plt.savefig('pyc_stefan_third_order_PSD_sizes_by_assembly.png')

# print('p-values', p_values)
# adjusted_p_values = smm.fdrcorrection(p_values, alpha=0.05, method='indep')[1]
# print('FDR p-values', adjusted_p_values)


# for chain in tqdm(four_chain_results):
#     origin_cell_in_pyc_id, middle_cell_in_pyc_id, middle_cell_two_in_pyc_id, terminal_cell_in_pyc_id = chain['A'], chain['B'], chain['C'], chain['D']

#     # Get the connectome (out of all cells, not just PYC) and root IDs for origin, middle, and terminal chain cells
#     origin_cell_in_connectome_id = cell_table.iloc[origin_cell_in_pyc_id].connectome_index
#     origin_cell_in_root_id = cell_table.iloc[origin_cell_in_pyc_id].pt_root_id

#     middle_cell_in_connectome_id = cell_table.iloc[middle_cell_in_pyc_id].connectome_index
#     middle_cell_in_root_id = cell_table.iloc[middle_cell_in_pyc_id].pt_root_id

#     middle_cell_two_in_connectome_id = cell_table.iloc[middle_cell_two_in_pyc_id].connectome_index
#     middle_cell_two_in_root_id = cell_table.iloc[middle_cell_two_in_pyc_id].pt_root_id
    
#     terminal_cell_in_connectome_id = cell_table.iloc[terminal_cell_in_pyc_id].connectome_index
#     terminal_cell_in_root_id = cell_table.iloc[terminal_cell_in_pyc_id].pt_root_id

#     origin_assembly_ids = None
#     terminal_assembly_ids = None
    
#     if (origin_cell_in_connectome_id in coregistered_unit_ids) and (terminal_cell_in_connectome_id in coregistered_unit_ids):
#         motifs_with_origin_and_terminal_coregistered += 1

#         origin_assembly_ids_temp = list(coregistered_cells.loc[coregistered_cells['unit_id'] == origin_cell_in_connectome_id]['assembly_membership'])
#         if len(origin_assembly_ids_temp) > 0:
#             origin_assembly_ids = list(origin_assembly_ids_temp[0])
#         terminal_assembly_ids_temp = list(coregistered_cells.loc[coregistered_cells['unit_id'] == terminal_cell_in_connectome_id]['assembly_membership'])
#         if len(terminal_assembly_ids_temp) > 0:
#             terminal_assembly_ids = list(terminal_assembly_ids_temp[0])

#         psd_sizes_origin_to_mid = list(synapse_table.query('pre_pt_root_id == @origin_cell_in_root_id and post_pt_root_id == @middle_cell_in_root_id')['size'].values)
#         psd_sizes_mid_to_second_mid = list(synapse_table.query('pre_pt_root_id == @middle_cell_in_root_id and post_pt_root_id == @middle_cell_two_in_root_id')['size'].values)
#         psd_sizes_second_mid_to_terminal = list(synapse_table.query('pre_pt_root_id == @middle_cell_two_in_root_id and post_pt_root_id == @terminal_cell_in_root_id')['size'].values)

#         if len(psd_sizes_origin_to_mid) == 0 or len(psd_sizes_second_mid_to_terminal) == 0: # If we for some reason don't have chains that connect to each other?!
#             nonconnected_motifs += 1
#             print("\n\tHUGE ERROR!  CHAIN CELLS AREN'T CONNECTED!!!!\n")
#         else:
#             # Do all combinations of assembly IDs, just for this first option.
#             if len(origin_assembly_ids) > 0 and len(terminal_assembly_ids) > 0:  # If we can identify assemblies for both
#                 # Add to pooled assemblies
#                 for size in psd_sizes_origin_to_mid:
#                     psd_sizes_pooled_assemblies.append(size)
#                 for size in psd_sizes_mid_to_second_mid:
#                     psd_sizes_pooled_assemblies.append(size) 
#                 for size in psd_sizes_second_mid_to_terminal:
#                     psd_sizes_pooled_assemblies.append(size)
#                 # Then add to each individual matched combo between origin and terminal assemblies
#                 for temp_origin_assembly_id in origin_assembly_ids:
#                     for temp_terminal_assembly_id in terminal_assembly_ids:
#                         if temp_origin_assembly_id == temp_terminal_assembly_id: # If they're both in the same assembly
#                             # Add to specific assembly
#                             interior_chains_by_assembly[f'Assembly {temp_origin_assembly_id+1}']
#                             for size in psd_sizes_origin_to_mid:
#                                 psd_size_list_by_assembly[f'Assembly {temp_origin_assembly_id+1}'].append(size)
#                             for size in psd_sizes_mid_to_second_mid:
#                                 psd_size_list_by_assembly[f'Assembly {temp_origin_assembly_id+1}'].append(size)
#                             for size in psd_sizes_second_mid_to_terminal:
#                                 psd_size_list_by_assembly[f'Assembly {temp_origin_assembly_id+1}'].append(size)
#                             interior_chains_by_assembly[f'Assembly {temp_origin_assembly_id+1}'] += 1

#         # For the rest, we rely just on the first ones.  If something belongs to multiple assemblies, it doesn't matter for these pools.
#             if len(origin_assembly_ids) == 0 and len(terminal_assembly_ids) == 0: # If neither is in any assembly
#                 if cell_table.iloc[origin_cell_in_pyc_id].soma_layer in ['23', '4']: # If this is a chain from 2/3/4 to 2/3/4, to allow chains to move through other layers, but compare ones that start and end among the same cell types as those in our functional assemblies from Session 1 Scan 3
#                     if cell_table.iloc[terminal_cell_in_pyc_id].soma_layer in ['23', '4']:
#                         for size in psd_sizes_origin_to_mid:
#                             psd_sizes_not_in_assemblies.append(size)
#                         for size in psd_sizes_mid_to_second_mid:
#                             psd_sizes_not_in_assemblies.append(size)
#                         for size in psd_sizes_second_mid_to_terminal:
#                             psd_sizes_not_in_assemblies.append(size)
#             if len(origin_assembly_ids) == 0 and len(terminal_assembly_ids) > 0: # If it crosses into an assembly
#                 if cell_table.iloc[origin_cell_in_pyc_id].soma_layer in ['23', '4']: # If this is a chain from 2/3/4
#                     for size in psd_sizes_origin_to_mid:
#                         psd_sizes_crossing_into_assemblies.append(size)
#                     for size in psd_sizes_mid_to_second_mid:
#                         psd_sizes_crossing_into_assemblies.append(size)
#                     for size in psd_sizes_second_mid_to_terminal:
#                         psd_sizes_crossing_into_assemblies.append(size)
#             if len(origin_assembly_ids) > 0 and len(terminal_assembly_ids) == 0: # If it crosses out of an assembly
#                 if cell_table.iloc[terminal_cell_in_pyc_id].soma_layer in ['23', '4']: # If it goes to 2/3/4
#                     for size in psd_sizes_origin_to_mid:
#                         psd_sizes_crossing_outof_assemblies.append(size)
#                     for size in psd_sizes_mid_to_second_mid:
#                         psd_sizes_crossing_into_assemblies.append(size)
#                     for size in psd_sizes_second_mid_to_terminal:
#                         psd_sizes_crossing_outof_assemblies.append(size)
#     else:
#         motifs_without_origin_and_terminal_coregistered += 1

# print("Nonconnected Chains: ", nonconnected_motifs)
# print("Motifs with start and end coregistered", motifs_with_origin_and_terminal_coregistered)
# print("Motifs with start and end not coregistered (excluded from analysis)", motifs_without_origin_and_terminal_coregistered)
# print("Motifs by assembly (start and end cells in the same assembly)", interior_chains_by_assembly)

# # Save work at this point
# with open('pyc_34chain_668_stefan_psd_sizes_by_assembly_dict.pickle', 'wb') as out_file:
#     pickle.dump(psd_size_list_by_assembly, out_file)

# np.save('pyc_34chain_668_stefan_psd_sizes_pooled_assemblies.npy', psd_sizes_pooled_assemblies)
# np.save('pyc_34chain_668_stefan_psd_sizes_not_in_assemblies.npy', psd_sizes_not_in_assemblies)
# np.save('pyc_34chain_668_stefan_psd_sizes_crossing_into_assemblies.npy', psd_sizes_crossing_into_assemblies)
# np.save('pyc_34chain_668_stefan_psd_sizes_crossing_outof_assemblies.npy', psd_sizes_crossing_outof_assemblies)

# # Plot and generate p-values
# p_values = [] 

# plt.figure(figsize=(30,10))
# legend_list = []

# non_assembly_sizes = np.squeeze(np.array(psd_sizes_not_in_assemblies)).flatten()
# plt.scatter([int(-20) for i in range(len(non_assembly_sizes))], list(non_assembly_sizes), c='k')
# print(f'Non-Assembly Mean Size:', np.mean(non_assembly_sizes))
# legend_list.append(f'Non-Assembly')
# sizes_array = np.squeeze(np.array(psd_sizes_pooled_assemblies)).flatten()
# plt.scatter([int(-10) for i in range(len(sizes_array))], list(sizes_array))
# print(f'Pooled Assemblies Mean Size:', np.mean(sizes_array))
# _, p_value = ranksums(non_assembly_sizes, sizes_array, alternative='less')
# print('Test: ', p_value)
# legend_list.append(f'Pooled Assemblies')

# for assembly in range(len(psd_size_list_by_assembly.keys())):
#     if len(psd_size_list_by_assembly[f'Assembly {assembly+1}']) > 0:
#         sizes_array = np.squeeze(np.array(psd_size_list_by_assembly[f'Assembly {assembly+1}'])).flatten()
#         # sizes_array = sizes_array
#         # print(sizes_array)
#         # bin_boundaries = np.arange(0,np.max(sizes_array)+np.max(sizes_array)/100,np.max(sizes_array)/100)
#         # print(len(sizes_by_assembly_dict[f'Assembly {assembly+1}']))
#         # plt.hist(sizes_by_assembly_dict[f'Assembly {assembly+1}'], bins=bin_boundaries, density=True)
#         plt.scatter([int(assembly) for i in range(len(sizes_array))], list(sizes_array))
#         print(f'Assembly {assembly+1} Mean Size:', np.mean(sizes_array))
#         _, p_value = ranksums(non_assembly_sizes, sizes_array, alternative='less')
#         p_values.append(p_value)
#         legend_list.append(f'Assembly {assembly+1}')



# plt.legend(legend_list)
# plt.title('Third-Order Structural PSD sizes by Functional Assembly Groups')
# plt.savefig('pyc_34chain_stefan_third_order_PSD_sizes_by_assembly.png')

# print('p-values', p_values)
# adjusted_p_values = smm.fdrcorrection(p_values, alpha=0.05, method='indep')[1]
# print('FDR p-values', adjusted_p_values)

