import pandas as pd
from kneed import KneeLocator
from neuprint import Client
from neuprint import fetch_adjacencies, NeuronCriteria as NC
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pre_HP 
from pathlib import Path
import pickle
import reliability_filter
import Hub_analysis


def input_parent(parent_list, x_cutoff, y_cutoff):
    """
    Create bipartite input DataFrames per parent cell type.
    Uses reliability_filter.frac_cell_pcnt_cutoff_v3('input', parent_name, x_cutoff=3, y_cutoff=0.8) for cutoff
    Has rank and reliability
    
    Parameters:
       parent_listall_celltypes_list_greater_or_eq_10 : a list of celltypes that have more than 10 cells in ct (~270).
    
    Returns:
        table: table mapping parent cell types to ranked bipartite DataFrames.
        df we are dealing with is a LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or partner ct syn information)
    """

    input_bipartite_table = []

    for parent_name in parent_list: 

        df_cutoff, df = reliability_filter.frac_cell_pcnt_cutoff_v3('input', parent_name, x_cutoff, y_cutoff) #df_cutoff is already filtered to only include partners that satisfy criteria, df is all partners
        
        # Cleaning
        df['raw_syn_nern'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('median')
        df['std_raw_syn_nern'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('std')
        df['range_raw_syn_nern'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('max') - df.groupby('type_pre')['sum_typepre_to_postId'].transform('min')

        df = df.drop(columns= ['bodyId_post', 'type_post', 'pcnt_ct_total_ct', 'num_neurons_top_ct','weighted_pcnt_ct_total_ct', 'median_pcnt_ct_total_ct', 'sum_weight_by_postId', 'sum_typepre_to_postId', 'roi', 'weight', 'pcnt_indv_postsyn_total_postsyn', 'num_typepres_to_postId', 'num_neurons_top_ct_missing'])
        df = df.drop_duplicates(subset=['type_pre', 'weight_median_pcnt_ct_total_ct'], keep='first')  

        # Rank
        sort_partners = df.groupby('type_pre')['weight_median_pcnt_ct_total_ct'].median().sort_values(ascending=False).index.tolist()
        rank_partners = pd.Series(range(1, len(sort_partners)+1), index=sort_partners)
        df['parent_rank'] = df['type_pre'].map(rank_partners)
        df = df.sort_values(by='parent_rank', ascending=True).reset_index(drop=True)

        df['parent'] = parent_name
        df = df.rename(columns={"type_pre": "partner_cell" , 
                                "weight_median_pcnt_ct_total_ct" : "pcnt_input_to_parent", 
                                'frac_parent_neurons_top_ct_seen' : 'partner_reliability', 
                                'med_syn_typepre_to_post': 'raw_syn', 
                                'std_syn_typepre_to_post': 'std_raw_syn',
                                'range_syn_typepre_to_post': 'range_raw_syn',
                                'avg_num_typepres_to_postId': 'avg_num_partners_contact_parent',
                                'med_num_typepres_to_postId': 'med_num_partners_contact_parent'})

        # Flag partner cells in cutoff for df (all partners df)
        cutoff_types = set(df_cutoff['type_pre'])
        df['cutoff'] = df['partner_cell'].isin(cutoff_types).astype(int)

        # append
        input_bipartite_table.append(df)
                
    input_bipartite_table_fin =  pd.concat(input_bipartite_table, ignore_index=True)
                
    return input_bipartite_table_fin

def output_parent(parent_list, x_cutoff, y_cutoff):
    """
    Create bipartite output DataFrames per parent cell type.
    Uses reliability_filter.frac_cell_pcnt_cutoff_v3('input', parent_name, x_cutoff=3, y_cutoff=0.8) for cutoff
    Has rank and reliability
    
    Parameters:
       parent_listall_celltypes_list_greater_or_eq_10 : a list of celltypes that have more than 10 cells in ct (~270).
    
    Returns:
        table: table mapping parent cell types to ranked bipartite DataFrames.
        df we are dealing with is a LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or partner ct syn information)
    """

    output_bipartite_table = []

    for parent_name in parent_list: 

        df_cutoff, df = reliability_filter.frac_cell_pcnt_cutoff_v3('output', parent_name, x_cutoff, y_cutoff) #df_cutoff is already filtered to only include partners that satisfy criteria, df is all partners
        
        # Cleaning
        df['raw_syn_nern'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('median')
        df['std_raw_syn_nern'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('std')
        df['range_raw_syn_nern'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('max') - df.groupby('type_post')['sum_typepost_from_preId'].transform('min')

        df = df.drop(columns= ['bodyId_pre', 'type_pre', 'pcnt_ct_total_ct', 'num_neurons_top_ct','weighted_pcnt_ct_total_ct', 'median_pcnt_ct_total_ct', "sum_weight_by_preId",'sum_typepost_from_preId', 'roi', 'weight', 'pcnt_indv_presyn_total_presyn', 'num_typeposts_from_preId', 'num_neurons_top_ct_missing'])
        df = df.drop_duplicates(subset=['type_post', 'weight_median_pcnt_ct_total_ct'], keep='first')  

        # Rank
        sort_partners = df.groupby('type_post')['weight_median_pcnt_ct_total_ct'].median().sort_values(ascending=False).index.tolist()
        rank_partners = pd.Series(range(1, len(sort_partners)+1), index=sort_partners)
        df['parent_rank'] = df['type_post'].map(rank_partners)
        df = df.sort_values(by='parent_rank', ascending=True).reset_index(drop=True)

        df['parent'] = parent_name
        df = df.rename(columns={"type_post": "partner_cell" , 
                                "weight_median_pcnt_ct_total_ct" : "pcnt_output_from_parent" ,
                                'frac_parent_neurons_top_ct_seen' : 'partner_reliability',
                                "med_syn_typepost_from_pre" : 'raw_syn',
                                'std_syn_typepost_from_pre': 'std_raw_syn',
                                'range_syn_typepost_from_pre': 'range_raw_syn',
                                'avg_num_typeposts_to_preId': 'avg_num_partners_contact_parent',
                                'med_num_typeposts_to_preId': 'med_num_partners_contact_parent'})

        # Flag partner cells in cutoff for df (all partners df)
        cutoff_types = set(df_cutoff['type_post'])
        df['cutoff'] = df['partner_cell'].isin(cutoff_types).astype(int)

        # append
        output_bipartite_table.append(df)

    output_bipartite_table_fin = pd.concat(output_bipartite_table, ignore_index=True)
                
    return output_bipartite_table_fin


def input_partner_to_output_parent(output_monopartite_table_v3):
    """
    Find partner to def output_parent(parent_list):
    Go to directory data_dir / partner_folder/ "input_" + i + "_df.pkl" 

    Parameters:
    - output_monopartite_table_v3: parents and their partners

    Returns:
        1. df with inputs from parents (ranked) and outputs from partners to parent (ranked based on partner connection to parent out of all partners) 
            df we are dealing with is a LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or partner ct syn information)
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    partner_folder = "partner_data_28_04_2026"

    bipartites = []
    output_monopartite_table_v3 = output_monopartite_table_v3[output_monopartite_table_v3['cutoff'] == 1]
    all_in_partners = set(output_monopartite_table_v3['partner_cell'])

    for i in all_in_partners: 
        # Open partner file
        filename = f"input_{i}_df.pkl"
        filepath = data_dir / partner_folder / filename
        

        with open(filepath, "rb") as f:
            input_df = pickle.load(f)

        _, _, _, input_to_celltype_df = pre_HP.get_input_connections_processing(input_df)
        df = input_to_celltype_df.copy()
        df['raw_syn_nern_partner_2'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('median')
        df['std_raw_syn_nern_partner_2'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('std')
        df['range_raw_syn_nern_partner_2'] = df.groupby('type_pre')['sum_typepre_to_postId'].transform('max') - df.groupby('type_pre')['sum_typepre_to_postId'].transform('min')

        df = df.drop(columns=['bodyId_post', 'type_post', 'pcnt_ct_total_ct', 'num_neurons_top_ct','weighted_pcnt_ct_total_ct', 'median_pcnt_ct_total_ct', 'sum_weight_by_postId', 'sum_typepre_to_postId','roi', 'weight', 'pcnt_indv_postsyn_total_postsyn', 'num_typepres_to_postId', 'num_neurons_top_ct_missing'])
        df = df.drop_duplicates(subset=['type_pre', 'weight_median_pcnt_ct_total_ct'], keep='first') 

        # rank 
        sort_partners = df.groupby('type_pre')['weight_median_pcnt_ct_total_ct'].median().sort_values(ascending=False).index.tolist()
        rank_partners = pd.Series(range(1, len(sort_partners)+1), index=sort_partners)
        df['partner_rank_2'] = df['type_pre'].map(rank_partners)
        df['partner_cell_2'] = i #need to add partner name in again as type_post deleted
        df = df.rename(columns={"type_pre": "parent_c", 
                                "weight_median_pcnt_ct_total_ct" : "pcnt_input_to_partner_2", 
                                "num_neurons_parent_ct": "total_partner_neurons_2", 
                                'frac_parent_neurons_top_ct_seen' : 'parent_reliability_2',
                                "med_syn_typepre_to_post": "raw_syn_partner_2",
                                'std_syn_typepre_to_post': 'std_raw_syn_partner_2',
                                'range_syn_typepre_to_post': 'range_raw_syn_partner_2',
                                'avg_num_typepres_to_postId': 'avg_num_parents_contact_partner_2',
                                'med_num_typepres_to_postId': 'med_num_parents_contact_partner_2'})

        #  Filter for parents with this partner
        bipartite_partner_downstream = output_monopartite_table_v3[output_monopartite_table_v3['partner_cell'] == i]
        parent = bipartite_partner_downstream['parent'].unique()
        for j in parent:
            parent_info_1 = bipartite_partner_downstream[bipartite_partner_downstream['parent'] == j]
            partner_info_2 = df[df['parent_c'] == j]
            merge =  partner_info_2.merge(parent_info_1, left_on='parent_c',right_on='parent', how='outer', sort=False)
            bipartites.append(merge)

    bipartites_final = pd.concat(bipartites, ignore_index=True)
    bipartites_final = bipartites_final.drop(columns= ['parent_c', "partner_cell_2"])
    bipartites_final = bipartites_final.sort_values(by = 'parent_rank', ascending=True).reset_index(drop=True)

    return bipartites_final

def output_partner_to_input_parent(input_monopartite_table_v3):
    """
    Find partner to def input_monopartite(parent_list):
    Go to directory data_dir / partner_folder "output_" + i + "_df.pkl" 

    Parameters:
    - input_monopartite_table_v3: parents and their partners

    Returns:
        1. df with inputs from parents (ranked) and outputs from partners to parent (ranked based on partner connection to parent out of all partners) 
            df we are dealing with is a LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or partner ct syn information)
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    partner_folder = "partner_data_28_04_2026"

    bipartites = []
    input_monopartite_table_v3 = input_monopartite_table_v3[input_monopartite_table_v3['cutoff'] == 1]
    all_out_partners = set(input_monopartite_table_v3['partner_cell'])


    for i in all_out_partners: 
        # Open partner file
        filename = f"output_{i}_df.pkl"
        filepath = data_dir / partner_folder / filename

        bipartite_partner_upstream = input_monopartite_table_v3[input_monopartite_table_v3['partner_cell'] == i]
        parent = bipartite_partner_upstream['parent'].unique()

        if filepath.exists():
            with open(filepath, "rb") as f:
                output_df = pickle.load(f)

            _, _, _, output_to_celltype_df = pre_HP.get_output_connections_processing(output_df)
            df = output_to_celltype_df.copy()
            df['raw_syn_nern_partner_2'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('median')
            df['std_raw_syn_nern_partner_2'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('std')
            df['range_raw_syn_nern_partner_2'] = df.groupby('type_post')['sum_typepost_from_preId'].transform('max') - df.groupby('type_post')['sum_typepost_from_preId'].transform('min')

            df = df.drop(columns= ['bodyId_pre','type_pre', 'pcnt_ct_total_ct', 'num_neurons_top_ct','weighted_pcnt_ct_total_ct', 'median_pcnt_ct_total_ct', 'sum_weight_by_preId', 'sum_typepost_from_preId', 'roi', 'weight', 'num_typeposts_from_preId', 'pcnt_indv_presyn_total_presyn', 'num_neurons_top_ct_missing'])
            df = df.drop_duplicates(subset=['type_post', 'weight_median_pcnt_ct_total_ct'], keep='first') 

            # rank 
            sort_partners = df.groupby('type_post')['weight_median_pcnt_ct_total_ct'].median().sort_values(ascending=False).index.tolist()
            rank_partners = pd.Series(range(1, len(sort_partners)+1), index=sort_partners)
            df['partner_rank_2'] = df['type_post'].map(rank_partners)
            df['partner_cell_2'] = i #need to add partner name in again as type_post deleted
            df = df.rename(columns={"type_post": "parent_c", 
                                    "weight_median_pcnt_ct_total_ct" : "pcnt_output_from_partner_2", 
                                    "num_neurons_parent_ct": "total_partner_neurons_2", 
                                    'frac_parent_neurons_top_ct_seen' : 'parent_reliability_2',
                                    "med_syn_typepost_from_pre": "raw_syn_partner_2",
                                    'std_syn_typepost_from_pre': 'std_raw_syn_partner_2',
                                    'range_syn_typepost_from_pre': 'range_raw_syn_partner_2',
                                    'avg_num_typeposts_to_preId': 'avg_num_parents_contact_partner_2',
                                    'med_num_typeposts_to_preId': 'med_num_parents_contact_partner_2'})                                      

            #  Filter for parents with this partner
            for j in parent:
                parent_info_1 = bipartite_partner_upstream[bipartite_partner_upstream['parent'] == j]
                partner_info_2 = df[df['parent_c'] == j]
                merge =  partner_info_2.merge(parent_info_1, left_on='parent_c',right_on='parent', how='outer', sort=False)
                bipartites.append(merge)

        else:
            for j in parent:

                parent_info_1 = bipartite_partner_upstream[bipartite_partner_upstream['parent'] == j]
                partner_info_2 = pd.DataFrame({
                    "parent_c": [j],
                    "pcnt_output_from_partner_2": [0],
                    "total_partner_neurons_2": [0],
                    "parent_reliability_2": [0],
                    "raw_syn_partner": [0],
                    'raw_syn_nern_partner': [0],
                    "partner_rank_2": [0],
                    "partner_cell_2": [i]
                })

                merge = partner_info_2.merge(
                    parent_info_1,
                    left_on='parent_c',
                    right_on='parent',
                    how='outer',
                    sort=False
                )

                bipartites.append(merge)


    bipartites_final = pd.concat(bipartites, ignore_index=True)
    bipartites_final = bipartites_final.drop(columns= ['parent_c', "partner_cell_2"])
    bipartites_final = bipartites_final.sort_values(by = 'parent_rank', ascending=True).reset_index(drop=True)

    return bipartites_final


def monopartite_population_first(all_celltypes_list_greater_or_eq_10, x_cutoff, y_cutoff):
    """
    Get all input and output to a parent.

    Arguments: all_celltypes_list_greater_or_eq_10

    Return: 
    - input_monopartite_table_v3
    - output_monopartite_table_v3

    """
    output_monopartite_table_v3 = output_parent(all_celltypes_list_greater_or_eq_10, x_cutoff, y_cutoff)
    input_monopartite_table_v3 = input_parent(all_celltypes_list_greater_or_eq_10, x_cutoff, y_cutoff)

    return input_monopartite_table_v3, output_monopartite_table_v3

def query_partners(input_monopartite_table_v3, output_monopartite_table_v3, all_ol_ct, all_celltypes_R_list ):
    """" 
    Query all partners.

    Arguments:
    - input_monopartite_table_v3: 
    - output_monopartite_table_v3:
    - all_ol_ct: from supplmentary figure 
    - all_ol_ct_R: right side only celltypes

    Returns:cha
    - 
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])

    failed_neurons = []   # collect neurons that fail

    #  ONLY COnsider partners BEFORE CUTOFF
    input_monopartite_table_v3_cutoff = input_monopartite_table_v3[input_monopartite_table_v3['cutoff'] == 1]
    output_monopartite_table_v3_cutoff = output_monopartite_table_v3[output_monopartite_table_v3['cutoff'] == 1]

    #  This is to see how many partners are non OL neurons or OL neurons < 10
    all_partners = set(input_monopartite_table_v3_cutoff['partner_cell']).union(set(output_monopartite_table_v3_cutoff['partner_cell']))#4368 all partners 
    
    # OL_cts_weak: partners not in original parent list (~270 cts. Query input and output).
    left_cell_body_partner = ((set(all_ol_ct['cell_type'])- set(all_celltypes_R_list)) & (set(all_partners))) #these are partners that are on the left side cell body (will crash on my code) (51)
    # OL_cts_r = (set(all_partners) & set(all_ol_ct_R['cell_type']))  #these are partners that are on the right side cell body and present as partners. (408 + 273) 
    # ## NOTE: I removed OL_cts_r because I want to query all right side OL neurons

    # CNS_partner_list: input side only for CNS partners
    CNS_partner_list = (set(all_partners) - set(all_ol_ct['cell_type'])) # all_ol_ct has 732 ct = 408 + 51 + 273, all partners = 4368

    partner_folder = "partner_data_28_04_2026"

    for i in CNS_partner_list:
        try:
            og_input_to_celltype_df = pre_HP.get_input_connections(i, 'right')
            
            with open(data_dir / partner_folder / f"input_{i}_df.pkl", "wb") as f:
                pickle.dump(og_input_to_celltype_df, f)

        except Exception as e:
            print(f"Failed querying CNS neuron: {i}")
            failed_neurons.append(i)

    for i in left_cell_body_partner:   
        og_input_to_celltype_df = pre_HP.get_input_connections(i, 'left')
        og_output_to_celltype_df = pre_HP.get_output_connections(i, 'left')

        with open(data_dir / partner_folder/ ("input_" + i + "_df.pkl"), "wb") as f:
            pickle.dump(og_input_to_celltype_df, f)

        with open(data_dir / partner_folder/ ("output_" + i + "_df.pkl"), "wb") as f:
            pickle.dump(og_output_to_celltype_df, f)

    if failed_neurons:
        failure_file = data_dir / partner_folder / "failed_neurons.txt"
        
        with open(failure_file, "a") as f:   # append mode
            for neuron in failed_neurons:
                f.write(neuron + "\n")

        print(f"{len(failed_neurons)} neurons failed and were logged.")
    else:
        print("No neurons failed.")

        # ["OA-VUMa4", "OA-VUMa8", "OA-VUMa1", "OA-VUMa3"] were  qwueried searately as in the middle

def bipartite_population_second(input_monopartite_table_v3, output_monopartite_table_v3):
    """
    Find bipartite pair for population. This is the second part of the pair (partner perspective to parent) 
    partner cell and its strength to parent with NO consideration of reliability cutoff to take into account. Parent may not be present in reliability based # of partner

    Parameters:
    - input_monopartite_table_v3: input to parent w reliability cutoff # of partners. 
    - output_monopartite_table_v3: output from parent w reliability cutoff # of partners. 

    Returns:
    bipartite_input_parent_output_partner : handshake information for a parent type. Input to parent, output from partner
    bipartite_output_parent_input_partner : handshake information for a parent type. output from parent, input to partner
    """

    bipartite_input_partner_output_parent = input_partner_to_output_parent(output_monopartite_table_v3)
    bipartite_output_partner_input_parent = output_partner_to_input_parent(input_monopartite_table_v3)

    return  bipartite_input_partner_output_parent, bipartite_output_partner_input_parent

# Bipartite strength
def bipartite_strength_df(all_celltypes_list_greater_or_eq_10, cut_bipartite_parent_input_v3, cut_bipartite_parent_output_v3):
    
    ########### ########### Strength Edges ########### ########### 
    # input df: partner -> parent
    input_edges = cut_bipartite_parent_input_v3.copy()

    input_edges["sender"] = input_edges["partner_cell"]
    input_edges["receiver"] = input_edges["parent"]

    input_edges["log_sender_over_receiver"] = np.log10(input_edges["total_partner_neurons_2"] / input_edges["num_neurons_parent_ct"])
    
    input_edges = input_edges.rename(columns={
        "pcnt_input_to_parent": "input_strength",
        "pcnt_output_from_partner_2": "output_strength",
        "total_partner_neurons_2": 'num_neurons_sender_ct',
        "num_neurons_parent_ct": 'num_neurons_receiver_ct',
        "raw_syn_nern_partner_2": 'nern_syn_sender', 
        "raw_syn_nern": 'nern_syn_receiver'
    })

    input_edges = input_edges[["sender", "receiver", "input_strength", "output_strength", 
                               "log_sender_over_receiver", 'num_neurons_sender_ct', 'num_neurons_receiver_ct', 'nern_syn_sender', 'nern_syn_receiver']]

    # output df: parent -> partner
    output_edges = cut_bipartite_parent_output_v3.copy()

    output_edges["sender"] = output_edges["parent"]
    output_edges["receiver"] = output_edges["partner_cell"]

    output_edges["log_sender_over_receiver"] = np.log10(output_edges["num_neurons_parent_ct"] / output_edges["total_partner_neurons_2"])

    output_edges = output_edges.rename(columns={
        "pcnt_input_to_partner_2": "input_strength",
        "pcnt_output_from_parent": "output_strength",
        "num_neurons_parent_ct": 'num_neurons_sender_ct',
        "total_partner_neurons_2": 'num_neurons_receiver_ct',
        "raw_syn_nern_partner_2": 'nern_syn_receiver', 
        "raw_syn_nern": 'nern_syn_sender'
    })

    output_edges = output_edges[["sender", "receiver", "input_strength", "output_strength", "log_sender_over_receiver",
                                  'num_neurons_sender_ct', 'num_neurons_receiver_ct', 'nern_syn_sender', 'nern_syn_receiver']]

    strength_edges = pd.concat([input_edges, output_edges], axis=0)
    strength_edges = strength_edges.drop_duplicates(
        subset=["sender", "receiver"],
        keep="first"
    )
    # Geometric Mean 
    strength_edges["geometric_mean_ct_pop"] = np.sqrt(strength_edges["num_neurons_sender_ct"] * strength_edges["num_neurons_receiver_ct"])
    # Number of partners 
    strength_edges['n_receiver_partners'] = strength_edges.groupby('receiver').transform('size')
    strength_edges['n_sender_partners'] = strength_edges.groupby('sender').transform('size')

    ########### ########### Hub Edges (hub-focals) ########### ########### 
    merged_partner_df = Hub_analysis.top_partners_freq_in_and_output_df(all_celltypes_list_greater_or_eq_10, 'Y', 3, 0.8)
    hub_out_df, hub_in_df, _, _ = Hub_analysis.parent_hub_df(merged_partner_df, all_celltypes_list_greater_or_eq_10, 'reliability')
    
    # input df: partner --> parent 
    hub_in_df["sender"] = hub_in_df["partner_cell"]
    hub_in_df["receiver"] = hub_in_df["parent"]

    hub_in_df = hub_in_df.rename(columns={
        "pcnt_input_to_parent": "input_strength",
        "pcnt_output_from_partner_2": "output_strength"
    })

    hub_in_df = hub_in_df[["sender", "receiver", "input_strength", "output_strength"]]

    # output df: parent -> partner
    hub_out_df["sender"] = hub_out_df["parent"]
    hub_out_df["receiver"] = hub_out_df["partner_cell"]

    hub_out_df = hub_out_df.rename(columns={
        "pcnt_input_to_partner_2": "input_strength",
        "pcnt_output_from_parent": "output_strength"})

    hub_out_df = hub_out_df[["sender", "receiver", "input_strength", "output_strength"]]

    hub_edges = pd.concat([hub_in_df, hub_out_df], axis=0)
    hub_edges = hub_edges.drop_duplicates(
        subset=["sender", "receiver"],
        keep="first"
    )

    #  reciprocal edges
    # Create a direction key for each edge
    strength_edges["edge_key"] = list(zip(strength_edges["sender"], strength_edges["receiver"]))
    # Create the reverse direction key
    strength_edges["reverse_key"] = list(zip(strength_edges["receiver"], strength_edges["sender"]))

    # Keep only edges whose reverse direction also exists
    all_edges = set(strength_edges["edge_key"])
    reciprocal_edges = strength_edges[(strength_edges["reverse_key"].isin(all_edges)) & (strength_edges["sender"] != strength_edges["receiver"])].copy()

    reciprocal_edges["reciprocal_id"] = [
        tuple(sorted((sender, receiver)))
        for sender, receiver in zip(
            reciprocal_edges["sender"],
            reciprocal_edges["receiver"]
        )
    ]
    #  New col
    reciprocal_edges['motif'] = 0
    for reciprocal_id, pair_df in reciprocal_edges.groupby("reciprocal_id", sort=False):
        count = ((pair_df[['input_strength', 'output_strength']] >= 10).values).sum()
        reciprocal_edges.loc[reciprocal_edges['reciprocal_id'] == reciprocal_id, 'motif'] = count
    

    return reciprocal_edges, strength_edges, hub_edges
def bipartite_fig_all(hub_neurons, strength_edges, reciprocal_edges, plot_type=None , cut_log_max=None, cut_log_min=None):
    #keep edges where either sender and receiver is above 10
    strength_edges = strength_edges[(strength_edges['input_strength'] >= 10) | (strength_edges['output_strength'] >= 10)].copy() 
    #remove edges where both sender and receiver below 10 cells in a cell type
    strength_edges = strength_edges[(strength_edges['num_neurons_sender_ct'] >= 10) & (strength_edges['num_neurons_receiver_ct'] >= 10)] 
    
    if plot_type == 'HP_only':
        # remove anything below 10 in strength and all reciprocals 
        strength_edges = strength_edges[(strength_edges['input_strength'] >= 10) & (strength_edges['output_strength'] >= 10)].copy()
        strength_edges = strength_edges[~strength_edges['reverse_key'].isin(reciprocal_edges['reverse_key'])] 
        strength_edges = strength_edges[~strength_edges['edge_key'].isin(reciprocal_edges['edge_key'])] 

    if plot_type == 'all_HP':
        strength_edges = strength_edges[(strength_edges['input_strength'] >= 10) & (strength_edges['output_strength'] >= 10)].copy()
    if cut_log_max is not None and cut_log_min is not None:
        strength_edges = strength_edges[(strength_edges["log_sender_over_receiver"] > cut_log_min) & (strength_edges["log_sender_over_receiver"] < cut_log_max)]

    # Marker style
    if plot_type == "log_sender_over_receiver":
        sizes = np.interp(np.log10(strength_edges["geometric_mean_ct_pop"] + 1), (np.log10(strength_edges["geometric_mean_ct_pop"].min() + 1), np.log10(strength_edges["geometric_mean_ct_pop"].max() + 1)), (5, 30))
        marker = dict(
        size=sizes,
        color=strength_edges["log_sender_over_receiver"],
        colorscale="RdYlBu",
        cmin=-3,
        cmax=3,
        showscale=True,
        colorbar=dict(
            title="Log (Sender / Receiver)",
            x=1.12,
            y=0.5,
            len=0.8,
        ),
    )
    else:
        marker = dict(
            size=12,
            color="blue",
            showscale=False,
        )


    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        row_heights=[0.2, 0.8],
        column_widths=[0.8, 0.2],
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )
    fig.add_trace(
        go.Scatter(
            x=strength_edges["input_strength"],
            y=strength_edges["output_strength"],
            text=strength_edges["sender"] + " → " + strength_edges["receiver"],
            customdata=strength_edges[[
                "input_strength",
                "output_strength",
                "num_neurons_sender_ct",
                "num_neurons_receiver_ct",
                "geometric_mean_ct_pop"
            ]],
            hovertemplate=(
                "%{text}<br>"
                "input_strength: %{customdata[0]:.3f}<br>"
                "output_strength: %{customdata[1]:.3f}<br>"
                "num_neurons_sender_ct: %{customdata[2]:.3f}<br>"
                "num_neurons_receiver_ct: %{customdata[3]:.3f}<br>"
                "geometric_mean_ct_pop: %{customdata[4]:.3f}"
                "<extra></extra>"
            ),
            mode="markers",
            marker=marker,
            name="Edges",
        ),
        row=2,
        col=1
    )

    # y=x line
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0,100],
            mode="lines",
            line=dict(
                color="white",
                dash="dash"
            ),
            name="y=x"
        ),
        row=2,
        col=1
    )

    # hub overlay
    if plot_type == "hub" or plot_type == "HP_only" or plot_type == "all_HP":
        hub_mask = (
            strength_edges["sender"].isin(hub_neurons["hub_neuron"]) |
            strength_edges["receiver"].isin(hub_neurons["hub_neuron"])
        )

        fig.add_trace(
            go.Scatter(
                x=strength_edges.loc[hub_mask, "input_strength"],
                y=strength_edges.loc[hub_mask, "output_strength"],
                text=(
                    strength_edges.loc[hub_mask, "sender"]
                    + " → "
                    + strength_edges.loc[hub_mask, "receiver"]
                ),
                hovertemplate=(
                    "%{text}<br>"
                    "Input strength: %{x:.3f}<br>"
                    "Output strength: %{y:.3f}"
                    "<extra></extra>"
                ),
                mode="markers",
                marker=dict(
                    size=12,
                    color="red"
                ),
                name="Hub edge"
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        width=1000,
        height=1000,
        template="plotly_dark",
        barmode="overlay",
        grid=None
    )

    fig.update_xaxes(title_text="Reciever / Input Strength", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="Sender /Output Strength", showgrid=False, row=2, col=1)

    fig.show()
def HP_fig_reciprocal(reciprocal_edges):
    motif_colors = {1: "purple", 2: "orange", 3: "blue", 4: "red"}
    reciprocal_edges = reciprocal_edges[(reciprocal_edges['num_neurons_sender_ct'] >= 10) & (reciprocal_edges['num_neurons_receiver_ct'] >= 10)] 
    fig = go.Figure()
    for motif in [1, 2, 3, 4]:#  remove reciprocal where all strengths are below 10  therefore 0 not included
        motif_df = reciprocal_edges[reciprocal_edges["motif"] == motif]

        first=True
        for _, pair_df in motif_df.groupby("reciprocal_id", sort=False):
            
            fig.add_trace(go.Scatter(
                x=pair_df["input_strength"],
                y=pair_df["output_strength"],
                text=(
                    pair_df["sender"]
                    + " → "
                    + pair_df["receiver"]
                ),
                hovertemplate=(
                    "%{text}<br>"
                    "Input strength: %{x:.3f}<br>"
                    "Output strength: %{y:.3f}"
                    "<extra></extra>"
                ),
                mode="lines+markers",
                # Dashed line connecting reciprocal directions
                line=dict(
                    color=motif_colors[motif],
                    width=2,
                    dash="dash"
                ),
                # White points with red circular outlines
                marker=dict(
                    size=16,
                    color=motif_colors[motif],
                ),

                name=f"Motif {motif}",
                legendgroup=f"Motif {motif}",
                showlegend=first,
            ))
            first = False

    fig.update_layout(
        width=1000,
        height=1000,
        xaxis_title="Input Strength",
        yaxis_title="Output Strength",
        template="simple_white",
        title=(
            "Reciprocal Edges"
        )
    )

    fig.show()
def plot_log_CT_size_HP_strength(strength_edges, hub_edges, strength_threshold=None, ct_num_threshold=None,main_color_axis=None,hub=False, hub_ct=None, other_ct=None):
    edges = strength_edges.copy()
    hub_edges = hub_edges.copy()
    hover_template = (
                    "<b>%{customdata[0]} → %{customdata[1]}</b><br>"
                    "Output strength: %{customdata[2]}<br>"
                    "Input strength: %{customdata[3]}<br>"
                    "Sender CT population: %{customdata[4]}<br>"
                    "Receiver CT population: %{customdata[5]}<br>"
                    "Sender synapses: %{customdata[6]}<br>"
                    "Receiver synapses: %{customdata[7]}<br>"
                    "Geometric mean population: %{customdata[8]}"
                    "<extra></extra>"
                )

    if strength_threshold is not None:
        edges = edges[(edges["input_strength"] >= strength_threshold) | (edges["output_strength"] >= strength_threshold)].copy()
    if ct_num_threshold is not None:
        edges = edges[(edges["num_neurons_sender_ct"] >= ct_num_threshold) & (edges["num_neurons_receiver_ct"] >= ct_num_threshold)].copy()
        
    edges["log_S_divide_R_strength"] = np.log10(edges["output_strength"] / edges["input_strength"])

    # Identify hub edges
    hub_pairs = set(zip(hub_edges["sender"], hub_edges["receiver"]))
    hub_mask = [(sender, receiver) in hub_pairs
        for sender, receiver in zip(edges["sender"],edges["receiver"])]
    hub_edge_df = edges.loc[hub_mask].copy()

    # Plot
    fig = go.Figure()
    # Plot all strength edges + highlighted hub edges
    
    fig.add_trace(
        go.Scatter(
            x=edges["log_sender_over_receiver"],
            y=edges["log_S_divide_R_strength"],
            mode="markers",
            name="Strength edges",
            marker=dict(
                size=7,
                color="lightgray",
                opacity=0.7,
                line=dict(width=0)),
            text=edges["sender"] + " → " + edges["receiver"],
            customdata=edges[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
            hovertemplate=hover_template
        )
    )
    if hub:
        fig.add_trace(
            go.Scatter(
                x=hub_edge_df["log_sender_over_receiver"],
                y=hub_edge_df["log_S_divide_R_strength"],
                mode="markers",
                name="Hub Edges",
                marker=dict(
                    size=10,
                    color="LightSkyBlue",
                    opacity=1,
                    line=dict(
                        color="MediumPurple",
                        width=3)),
                text=hub_edge_df["sender"] + " → " + hub_edge_df["receiver"],
                customdata=hub_edge_df[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
                hovertemplate=hover_template
            )
        )
        # Separate hub edges into:
        #   red = hub is sender
        #   blue = hub is receiver
        if hub_ct is not None:
            hub_sender = hub_edge_df[hub_edge_df["sender"] == hub_ct].copy()
            hub_receiver = hub_edge_df[hub_edge_df["receiver"] == hub_ct].copy()
            fig.add_trace(
                go.Scatter(
                    x=hub_sender["log_sender_over_receiver"],
                    y=hub_sender["log_S_divide_R_strength"],
                    mode="markers",
                    name="Hub as sender",
                    marker=dict(
                        size=12,
                        color="red",
                        opacity=1.0,
                        line=dict(
                            color="black",
                            width=1
                        )
                    ),
                    customdata=hub_sender[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
                    hovertemplate=hover_template
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hub_receiver["log_sender_over_receiver"],
                    y=hub_receiver["log_S_divide_R_strength"],
                    mode="markers",
                    name="Hub as receiver",
                    marker=dict(
                        size=12,
                        color="blue",
                        opacity=1.0,
                        line=dict(
                            color="black",
                            width=1
                        )
                    ),
                    customdata=hub_receiver[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
                    hovertemplate=hover_template))
            
    if other_ct is not None:
        other_sender = edges[edges["sender"] == other_ct].copy()
        other_receiver = edges[edges["receiver"] == other_ct].copy()
        fig.add_trace(
            go.Scatter(
                x=other_sender["log_sender_over_receiver"],
                y=other_sender["log_S_divide_R_strength"],
                mode="markers",
                name="CT as sender",
                marker=dict(
                    size=12,
                    color="red",
                    opacity=1.0,
                    line=dict(
                        color="black",
                        width=1
                    )
                ),
                customdata=other_sender[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
                hovertemplate=hover_template
            )
        )
        fig.add_trace(
            go.Scatter(
                x=other_receiver["log_sender_over_receiver"],
                y=other_receiver["log_S_divide_R_strength"],
                mode="markers",
                name="CT as receiver",
                marker=dict(
                    size=12,
                    color="blue",
                    opacity=1.0,
                    line=dict(
                        color="black",
                        width=1
                    )
                ),
                customdata=other_receiver[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
                hovertemplate=hover_template))
    # Only strength edges
    if main_color_axis is not None:
        range_color_min = edges[main_color_axis].min()
        range_color_max = edges[main_color_axis].max()
        fig.add_trace(
            go.Scatter(
                x=edges["log_sender_over_receiver"],
                y=edges["log_S_divide_R_strength"],
                mode="markers",
                name="Strength edges",
                marker=dict(
            size=8,
            color=edges[main_color_axis],
            colorscale="Sunsetdark",
            cmin=range_color_min,
            cmax=range_color_max,
            colorbar=dict(
                title=main_color_axis
            ),
            opacity=1.0),
            customdata=edges[["sender", "receiver", "output_strength", "input_strength", "num_neurons_sender_ct", "num_neurons_receiver_ct", "nern_syn_sender", "nern_syn_receiver", "geometric_mean_ct_pop"]].values,
            hovertemplate=hover_template
            )
        )
    if hub and hub_ct is not None:
        plot_title = f"Hub Cell Type: {hub_ct}"
    elif other_ct is not None:
        plot_title = f"Cell Type: {other_ct}"
    elif hub:
        plot_title = "Hub Edges"
    else:
        plot_title = "Cell Type-to-Cell Type Connections"

    fig.update_layout(
        width=1000,
        height=1000,
        font=dict(
            family="Arial",
            size=14
        ),
        template="plotly_white",
        title=dict(
            text=plot_title,
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top"
        )
    )

    fig.update_xaxes(
        title="Log (Sender / Receiver) Population Size",
        scaleanchor="y",
        scaleratio=1
    )

    fig.update_yaxes(
        title="Log (Sender / Receiver) Strength"
    )

    return fig  

def HP_quant(strength_edges, reciprocal_edges, hub_neurons):
    # ct greater than 10 and strength too (this is all HPs including reciprocals )
    strength_edges = strength_edges[(strength_edges['input_strength'] >= 10) & (strength_edges['output_strength'] >= 10)].copy() 
    strength_edges = strength_edges[(strength_edges['num_neurons_sender_ct'] >= 10) & (strength_edges['num_neurons_receiver_ct'] >= 10)] 

    # flag hubs, symmetric, unsym HPs
    strength_edges["hub_HP"] = (strength_edges["sender"].isin(hub_neurons["hub_neuron"]) | strength_edges["receiver"].isin(hub_neurons["hub_neuron"])).astype(int)
    strength_edges["symmetry"] = np.isclose(strength_edges["output_strength"], strength_edges["input_strength"], rtol=0.2).astype(int)

    # A pair is reciprocal if it appears in both directions
    reciprocal_edges_4 = reciprocal_edges[reciprocal_edges['motif'] == 4]
    HP_r = list(zip( reciprocal_edges_4['sender'], reciprocal_edges_4['receiver']))
    strength_edges["HP_r"] = (strength_edges["edge_key"].isin(HP_r) | strength_edges["reverse_key"].isin(HP_r)).astype(int)

    HP_quant_df = pd.DataFrame({
            "Category": [
                "HP_all",
                "HP_hub_focal",
                "HP_sym",
                "HP_r"
            ],

            "Count": [len(strength_edges) - len(reciprocal_edges_4), strength_edges["hub_HP"].sum(), strength_edges["symmetry"].sum(), strength_edges["HP_r"].sum()]})

    return HP_quant_df, strength_edges
def HP_hub_symmetry_matrix_plot(strength_edges_v2):
     # Cell-type pair as index
    strength_edges_v2  = strength_edges_v2.sort_values(by=["sender", "receiver"]).reset_index(drop=True)
    strength_edges_v2["cell_type_pair"] = (strength_edges_v2["sender"] + " → " + strength_edges_v2["receiver"])
    HP_matrix = strength_edges_v2[["cell_type_pair", "hub_HP", "HP_r", "symmetry"]].copy()
    HP_matrix = HP_matrix.set_index("cell_type_pair")
    fig = px.imshow(
        HP_matrix.T,
        aspect="auto",
        labels={
            "x": "Cell-type pair",
            "y": ""
        },
        color_continuous_scale=["white", "purple"],
        zmin=0,
        zmax=1
    )

    fig.update_layout(
        width=5000,
        height=600
    )

    return fig.show()