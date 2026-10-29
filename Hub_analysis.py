import pandas as pd
from kneed import KneeLocator
from neuprint import Client
from neuprint import fetch_adjacencies, NeuronCriteria as NC, fetch_synapses, SynapseCriteria as SC, fetch_neurons
import numpy as np
from kneed import KneeLocator
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pre_HP
import Handshake_function_v1
import reliability_filter
import inspect
import yaml
from pathlib import Path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from matplotlib.colors import LinearSegmentedColormap

# (A.1) Top 5 src partner plot (df and plot)
def top_ct_partner_freq_df(n, direction): 
    """
    Find number of partners to consider - pure cutoff method! This has parents: all_celltypes_list_greater_or_eq_10: alll celltypes w more or eq to 10 neurons 
    Arguments:
    - direction: input or output 
    - n = number of partners to include (eg: if '3' then will include 3 partners) 

    Note: when using 'Y' option for partner reliability keep in mind that the bipartite df I am uploading are already filtered for partner reliability to be above 0.8

    Return:
    df_fin: w pcnt, parent rank, reliability and partner info till cutoff 
    partner_rank_freq: how often a partner occurs within a bined rank (1,2,3,4,5+)
    rank_order: order by most occuring partner to least
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    # Saved dictionary of first part of bipartite pair (major parents)

    if direction == 'input':
        type = 'type_pre'
        to_drop = ["L1", "L2", "L3", "L4", "L5", "T1", "R1-R6"]
        pcnt = 'pcnt_input_to_parent'
        total_pcnt = 'total_partner_pcnt_input'
        med_pcnt = 'med_partner_pcnt_input'
        nern_syn = 'raw_syn_nern'
        med_syn = 'med_raw_syn_nern'
        with open(data_dir / "input_monopartite_table_v3.pkl", "rb") as f:
            df = pickle.load(f)

    elif direction == 'output':
        type = 'type_post'
        to_drop = ["R1-R6"]
        pcnt = 'pcnt_output_from_parent'
        total_pcnt = 'total_partner_pcnt_output'
        med_pcnt = 'med_partner_pcnt_output'
        nern_syn = 'raw_syn_nern'
        med_syn = 'med_raw_syn_nern'
        with open(data_dir / "output_monopartite_table_v3.pkl", "rb") as f:
            df = pickle.load(f)
    else:
        to_drop = []

    df = df[~df["parent"].isin(to_drop)]
    df_fin = df[df['cutoff'] == 1]
    df_fin = df_fin[df_fin["parent_rank"] <= n] # only include partners less than this cutoff

    df_fin[total_pcnt] = df_fin.groupby('partner_cell')[pcnt].transform('sum')
    df_fin[med_pcnt] = df_fin.groupby('partner_cell')[pcnt].transform('median')
    df_fin[med_syn] = df_fin.groupby('partner_cell')[nern_syn].transform('median')
    df_fin['parent_rank'] = df_fin['parent_rank'].astype(str)
    
    # Order by all top_ct
    partner_rank_freq = df_fin.groupby(['parent_rank', 'partner_cell']).size().reset_index(name='freq') # this will be used in stacked plot to generate stacks based on rank
    partner_freq = df_fin.groupby('partner_cell').size().reset_index(name='freq')# this is just for ordering x axis 
    rank_order = partner_freq.sort_values('freq', ascending=False)['partner_cell'].tolist()

    return df_fin, partner_rank_freq, rank_order   
def top_ct_partner_freq_plot(n, direction, df_fin, partner_rank_freq, rank_order): 
    """
    Find number of partners to consider (this is using hard-coded threhold of x partners). 

    Argument:
    df_fin: partners cutoff at n parameter
    partner_rank_freq: how often a partner occurs within a bined rank (1,2,3,4,5+)
    rank_order: order by most occuring partner to least

    Return: subplot
    - plot 1 is the total pcnt the partner contributes to vrious parents (one value per partner on x axis)
    - plot 2 is stacked plot with number of times the partner occurs as a parent's rank 1,2,3,4,5+
    """
    if direction == 'input':
        total_pcnt = 'total_partner_pcnt_input'
        pcnt = 'pcnt_input_to_parent'


    elif direction == 'output':
        total_pcnt = 'total_partner_pcnt_output'
        pcnt = 'pcnt_output_from_parent'
      
    # Plot
    bubble_plot = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Top CT Scatter with % they Connect with Parent", "Frequency Top CT Occurance Stacked by Rank"]
        )

    # bubble_plot.add_trace(
    #     go.Scatter( 
    #         x=df_fin['partner_cell'],
    #         y=df_fin[total_pcnt],
    #         text=df_fin['partner_cell'],
    #         hoverinfo='text',
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #         ),  
    #         showlegend=True       
    #     ),
    #     row=1, col=1
    #     )
    bubble_plot.add_trace(
        go.Box( 
            x=df_fin['partner_cell'],
            y=df_fin[pcnt],
            text=df_fin['partner_cell'],
            hoverinfo='text',
            showlegend=True       
        ),
        row=1, col=1
        )

    ranks = {'1': '#54278f' , '2': '#756bb1', '3':'#9e9ac8', '4':'#bcbddc' , '5':'#dadaeb' }
    for r, col in ranks.items():
        stacked_df = partner_rank_freq[partner_rank_freq['parent_rank'] == r]
        bubble_plot.add_trace(
            go.Bar(
                x=stacked_df['partner_cell'],
                y=stacked_df['freq'],
                name=r,
                marker=dict(color=col),
                legendgroup=r,
            ),
            row=2, col=1
        )
    for row in [1, 2]:
        bubble_plot.update_xaxes(categoryorder='array',categoryarray=rank_order,tickfont={'size': 10},row=row, col=1)
        bubble_plot.update_yaxes(title_text='Total % Partner Contribution', row=1, col=1)
        bubble_plot.update_yaxes(title_text='Frequency of Top CT Occurrence per Rank', row=2, col=1)
        
    bubble_plot.update_layout(barmode='stack',plot_bgcolor='white', title=f"Partner Frequency direction: {direction} with Cutoff Threshold:{n} partners")

    # Make histogram of all strenght of connections distribution in a direction (in percentage)
    hist_pcnt_dist = px.histogram(df_fin, x=pcnt)
    # fig.update_yaxes(type="log")

    return bubble_plot, hist_pcnt_dist
# (A.2) Plot top_ct 5 of parent as a funciton of their freq of occurance (both directions mapped together heatmap and scatter)
def hub_plot_topx_df(n):
    """ This to process data before plotting in input and output partner freq
    Arguments:
    - n: number of partners to include (eg: if '3' then will include 3 partners) 

    Return:
    merged_partner_num: number of time partner occurs in list of parents (merged in and output)
    merged_partner_weight: total sum of pcnt where partner is involved in a connection to a parent (the pcnt is from a parent's perspective) (merged in and output)

    """
    df_fin_input, partner_rank_freq_input, _ = top_ct_partner_freq_df(n, 'input')
    df_fin_output, partner_rank_freq_output, _ = top_ct_partner_freq_df(n, 'output')

    # Process frequency table
    partner_num_input = (
        partner_rank_freq_input
        .assign(freq_input_all_ranks = partner_rank_freq_input.groupby('partner_cell')['freq'].transform('sum'))
        .drop_duplicates(subset=['partner_cell', 'freq_input_all_ranks'])
        .drop(columns=['parent_rank' , 'freq'])
    )

    partner_num_output = (
        partner_rank_freq_output
        .assign(freq_output_all_ranks = partner_rank_freq_output.groupby('partner_cell')['freq'].transform('sum'))
        .drop_duplicates(subset=['partner_cell', 'freq_output_all_ranks'])
        .drop(columns=['parent_rank' , 'freq'])
    )

    # Process weight table
    drop_cols = [
        'raw_syn', 'num_neurons_parent_ct', 'partner_reliability', 'parent_rank', 'parent', 'cutoff', 'avg_num_partners_contact_parent', 'med_num_partners_contact_parent', 'raw_syn_nern'
    ]

    partner_weight_input = (df_fin_input
                            .drop(columns=['pcnt_input_to_parent' , *drop_cols])
                            .drop_duplicates(subset=['partner_cell', 'med_partner_pcnt_input'])
                            .rename(columns={'med_raw_syn_nern' : 'med_syn_nern_input'}))
    partner_weight_output = (df_fin_output
                            .drop(columns=['pcnt_output_from_parent' , *drop_cols])
                            .drop_duplicates(subset=['partner_cell', 'med_partner_pcnt_output'])
                            .rename(columns={'med_raw_syn_nern' : 'med_syn_nern_output'}))
    
    # # Perform Merge 
    merged_partner_num = partner_num_input.merge(partner_num_output, on='partner_cell', how='outer')
    merged_partner_weight = partner_weight_input.merge( partner_weight_output, on='partner_cell', how='outer')
    merged_partner_df = merged_partner_num.merge(
    merged_partner_weight,
    on='partner_cell',
    how='outer'
)

    empty_col_num = ['freq_input_all_ranks', 'freq_output_all_ranks', 'med_partner_pcnt_input', 'med_partner_pcnt_output', 'med_syn_nern_output', 'med_syn_nern_input']

    merged_partner_df[empty_col_num] = merged_partner_df[empty_col_num].fillna(0)

    return merged_partner_df
def hub_plot_topx_plot(merged_partner_df, n, direction_color):
    """ 
    Plot input and output plot 
    direction_color: color the plot by either input or output strength ('input' or 'output)
    """
    if direction_color == 'input':
        color_dot = 'med_partner_pcnt_input'
    if direction_color == 'output':
        color_dot = 'med_partner_pcnt_output'

    # Create masked color column
    merged_partner_df['color_masked'] = np.where(
        (merged_partner_df['freq_input_all_ranks'] >= 12) |
        (merged_partner_df['freq_output_all_ranks'] >= 11),
        merged_partner_df[color_dot],   # keep real values
        np.nan            # everything else becomes NaN
    )
    # Plot
    num_plot = px.scatter(
    merged_partner_df,
    x='freq_input_all_ranks',
    y='freq_output_all_ranks',
    color='color_masked',
    marginal_x='histogram',
    marginal_y='histogram',
    hover_data='partner_cell')
    
    num_plot.update_layout(plot_bgcolor='white', 
                           title=f"Partner Hub Plot (Number of Partners I/O) with Cutoff Threshold:{n} partners. Colored by {direction_color}",
                            xaxis_title='Input',
                            yaxis_title='Output',
                            xaxis2_title=None,
                            yaxis2_title=None)

    return num_plot


# (B.1) Reliability based cutoff SRC partner plot
def top_ct_partner_freq_reliability_df(all_celltypes_list_greater_or_eq_10, partner_reliability, direction, x_cutoff, y_cutoff):
    """
    Find number of partners to consider use reliability method: 

    Arguments:
    - all_celltypes_list_greater_or_eq_10: alll celltypes w more or eq to 10 neurons 
    - direction: input or output 
    - x_cutoff: pcnt cut off (e.g: 1, 2, 0.1) 
    - y_cutoff: reliability (e.g: 0.8, 0.1)
    - partner_reliability: include cutting off celltypes from partner perspective (less than 0.8 = Y) (Y/N)

    Note: when using 'Y' option for partner reliability keep in mind that the bipartite df I am uploading are already filtered for partner reliability to be above 0.8

    Return:
    df_fin: w pcnt, parent rank, reliability and partner info till cutoff 
    partner_rank_freq: how often a partner occurs within a bined rank (1,2,3,4,5+)
    rank_order: order by most occuring partner to least
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    # Saved dictionary of first part of bipartite pair (major parents)
    if partner_reliability == 'N':
        if direction == 'input':
            type = 'type_pre'
            to_drop = ["L1", "L2", "L3", "L4", "L5", "T1", "R1-R6"]
            pcnt = 'pcnt_input_to_parent'
            total_pcnt = 'total_partner_pcnt_input'
            med_pcnt = 'med_partner_pcnt_input'
            nern_syn = 'raw_syn_nern'
            med_syn = 'med_raw_syn_nern'
            with open(data_dir / "input_monopartite_table_v3.pkl", "rb") as f:
                df = pickle.load(f)

        elif direction == 'output':
            type = 'type_post'
            to_drop = ["R1-R6"]
            pcnt = 'pcnt_output_from_parent'
            total_pcnt = 'total_partner_pcnt_output'
            med_pcnt = 'med_partner_pcnt_output'
            nern_syn = 'raw_syn_nern'
            med_syn = 'med_raw_syn_nern'
            with open(data_dir / "output_monopartite_table_v3.pkl", "rb") as f:
                df = pickle.load(f)
        else:
            to_drop = []


        df = df[~df["parent"].isin(to_drop)]

        new_cutoff_table = []
        if (x_cutoff, y_cutoff) != (3, 0.8):
            for parent_name in all_celltypes_list_greater_or_eq_10:
                #  default x_cut = 3 and y_cut = 0.8 (see column [cutoff] where I have flagged this: 1 is above this threshold and 0 is below)
                df_cutoff, _ = reliability_filter.frac_cell_pcnt_cutoff_v3(direction, parent_name, x_cutoff=x_cutoff, y_cutoff=y_cutoff)
                cutoff_types = set(df_cutoff[type])
                new_cutoff = df[df['parent'] == parent_name].copy()
                new_cutoff['cutoff'] = new_cutoff['partner_cell'].isin(cutoff_types).astype(int)
                
                # append
                new_cutoff_table.append(new_cutoff)
                        
            df_fin =  pd.concat(new_cutoff_table, ignore_index=True)

        else:
            df_fin = df.copy()

        df_fin = df_fin[df_fin['cutoff'] == 1]

    elif partner_reliability == 'Y':
        if direction == 'input':
            to_drop = ["L1", "L2", "L3", "L4", "L5", "T1", "R1-R6"]
            pcnt = 'pcnt_input_to_parent'
            total_pcnt = 'total_partner_pcnt_input'
            med_pcnt = 'med_partner_pcnt_input'
            nern_syn = 'raw_syn_nern'
            med_syn = 'med_raw_syn_nern'
            with open(data_dir / "bipartite_input_parent_output_partner_v3.pkl", "rb") as f:
                df = pickle.load(f)

        elif direction == 'output':
            to_drop = ["R1-R6"]
            pcnt = 'pcnt_output_from_parent'
            total_pcnt = 'total_partner_pcnt_output'
            med_pcnt = 'med_partner_pcnt_output'
            nern_syn = 'raw_syn_nern'
            med_syn = 'med_raw_syn_nern'
            with open(data_dir / "bipartite_output_parent_input_partner_v3.pkl", "rb") as f:
                df = pickle.load(f)
        else:
            to_drop = []

        df = df[~df["parent"].isin(to_drop)]
        df  = df[df["parent_reliability_2"] >= 0.8]

        if (x_cutoff, y_cutoff) != (3, 0.8):
            new_cutoff_table = []
            grouped = df.groupby('parent')
            for parent_name, df_parent in grouped:
                # Escape criteria partners
                partners_escape_criteria = ['R7y', 'R7p', 'R8y', 'R8p']
                df_below_threshold = df_parent[
                    (~df_parent['partner_cell'].isin(partners_escape_criteria)) &   # ← protect R7/R8
                    (df_parent[pcnt] < x_cutoff) &
                    (df_parent['partner_reliability'] < y_cutoff)
                ].copy()

                partner_cells_remove = df_below_threshold['partner_cell'].unique().tolist()
                initial_cut = df_parent[~df_parent['partner_cell'].isin(partner_cells_remove)].copy()

                # remove 'R7y', 'R7p', 'R8y', 'R8p' below 1 pcnt and 0.4 reliability 
                remove_R7_R8_below_cutoff = (
                    initial_cut['partner_cell'].isin(partners_escape_criteria) &
                    (initial_cut[pcnt] < 1) &
                    (initial_cut['partner_reliability'] < 0.4)
                )
                new_cutoff = initial_cut[~remove_R7_R8_below_cutoff] # returns neuron ID to neuron ID level df
                new_cutoff_table.append(new_cutoff)
                        
            df_fin =  pd.concat(new_cutoff_table, ignore_index=True)
            # df_fin = df_fin[df_fin['parent_reliability_2'] >= 0.8]
        else:
            df_fin = df.copy()
            # df_fin = df_fin[df_fin['parent_reliability_2'] >= 0.8]

    df_fin[total_pcnt] = df_fin.groupby('partner_cell')[pcnt].transform('sum')
    df_fin[med_pcnt] = df_fin.groupby('partner_cell')[pcnt].transform('median')
    df_fin[med_syn] = df_fin.groupby('partner_cell')[nern_syn].transform('median')
    df_fin[f"{med_syn}_partner"] = df_fin.groupby('partner_cell')[f"{nern_syn}_partner_2"].transform('median')
    df_fin['parent_rank_group'] = (df_fin['parent_rank'].astype(str).where(df_fin['parent_rank'].isin([1, 2, 3, 4]), '5+'))
    df_fin['parent_rank_group'] = df_fin['parent_rank_group'].astype(str)
    
    # Order by all top_ct
    partner_rank_freq = df_fin.groupby(['parent_rank_group', 'partner_cell']).size().reset_index(name='freq') # this will be used in stacked plot to generate stacks based on rank
    partner_freq = df_fin.groupby('partner_cell').size().reset_index(name='freq')# this is just for ordering x axis 
    rank_order = partner_freq.sort_values('freq', ascending=False)['partner_cell'].tolist()

    return df_fin, partner_rank_freq, rank_order
def top_ct_partner_freq_reliability_plot(direction,df_fin, partner_rank_freq, rank_order,x_cutoff, y_cutoff):
    """
    Find number of partners to consider. 

    Argument:
    df_fin: w pcnt, parent rank, reliability and partner info till cutoff 
    partner_rank_freq: how often a partner occurs within a bined rank (1,2,3,4,5+)
    rank_order: order by most occuring partner to least

    Return: subplot
    - plot 1 is the total pcnt the partner contributes to vrious parents (one value per partner on x axis)
    - plot 2 is stacked plot with number of times the partner occurs as a parent's rank 1,2,3,4,5+
    """
    if direction == 'input':
        total_pcnt = 'total_partner_pcnt_input'
        pcnt = 'pcnt_input_to_parent'


    elif direction == 'output':
        total_pcnt = 'total_partner_pcnt_output'
        pcnt = 'pcnt_output_from_parent'
      
    # Plot
    bubble_plot = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=["Top CT Scatter with % they Connect with Parent", "Frequency Top CT Occurance Stacked by Rank"]
        )

    # bubble_plot.add_trace(
    #     go.Scatter( 
    #         x=df_fin['partner_cell'],
    #         y=df_fin[total_pcnt],
    #         text=df_fin['partner_cell'],
    #         hoverinfo='text',
    #         mode='markers',
    #         marker=dict(
    #             size=12,
    #         ),  
    #         showlegend=True       
    #     ),
    #     row=1, col=1
    #     )
    bubble_plot.add_trace(
        go.Box( 
            x=df_fin['partner_cell'],
            y=df_fin[pcnt],
            text=df_fin['partner_cell'],
            hoverinfo='text',
            showlegend=True       
        ),
        row=1, col=1
        )

    ranks = {'1': '#54278f' , '2': '#756bb1', '3':'#9e9ac8', '4':'#bcbddc' , '5+':'#dadaeb' }
    for r, col in ranks.items():
        stacked_df = partner_rank_freq[partner_rank_freq['parent_rank_group'] == r]
        bubble_plot.add_trace(
            go.Bar(
                x=stacked_df['partner_cell'],
                y=stacked_df['freq'],
                name=r,
                marker=dict(color=col),
                legendgroup=r,
            ),
            row=2, col=1
        )
    for row in [1, 2]:
        bubble_plot.update_xaxes(categoryorder='array',categoryarray=rank_order,tickfont={'size': 10},row=row, col=1)
        bubble_plot.update_yaxes(title_text='Total % Partner Contribution', row=1, col=1)
        bubble_plot.update_yaxes(title_text='Frequency of Top CT Occurrence per Rank', row=2, col=1)
        
    bubble_plot.update_layout(barmode='stack',plot_bgcolor='white', title=f"Partner Frequency direction: {direction} with Cutoff Thresholds %:{x_cutoff} Reliability:{y_cutoff} ")

    # Make histogram of all strenght of connections distribution in a direction (in percentage)
    hist_pcnt_dist = px.histogram(df_fin, x=pcnt)
    # fig.update_yaxes(type="log")


    return bubble_plot, hist_pcnt_dist

# (B.2) input and output hub plot with reliability 
def top_partners_freq_in_and_output_df(all_celltypes_list_greater_or_eq_10, partner_reliability, x_cutoff, y_cutoff):
    """ This to process data before plotting in input and output partner freq
    Arguments:
    - all_celltypes_list_greater_or_eq_10: alll celltypes w more or eq to 10 neurons 
    - x_cutoff: pcnt cut off (e.g: 1, 2, 0.1) 
    - y_cutoff: reliability (e.g: 0.8, 0.1)

    Return:
    merged_partner_num: number of time partner occurs in list of parents (merged in and output)
    merged_partner_weight: total sum of pcnt where partner is involved in a connection to a parent (the pcnt is from a parent's perspective) (merged in and output)

    """
    df_fin_input, partner_rank_freq_input, _ = top_ct_partner_freq_reliability_df(all_celltypes_list_greater_or_eq_10, partner_reliability, 'input', x_cutoff, y_cutoff)
    df_fin_output, partner_rank_freq_output, _ = top_ct_partner_freq_reliability_df(all_celltypes_list_greater_or_eq_10, partner_reliability, 'output', x_cutoff, y_cutoff)

    # Process frequency table
    partner_num_input = (
        partner_rank_freq_input
        .assign(freq_input_all_ranks = partner_rank_freq_input.groupby('partner_cell')['freq'].transform('sum'))
        .drop_duplicates(subset=['partner_cell', 'freq_input_all_ranks'])
        .drop(columns=['parent_rank_group' , 'freq'])
    )

    partner_num_output = (
        partner_rank_freq_output
        .assign(freq_output_all_ranks = partner_rank_freq_output.groupby('partner_cell')['freq'].transform('sum'))
        .drop_duplicates(subset=['partner_cell', 'freq_output_all_ranks'])
        .drop(columns=['parent_rank_group' , 'freq'])
    )

    # Process weight table
    drop_cols = [
        'raw_syn', 'num_neurons_parent_ct', 'partner_reliability', 'parent_rank', 'parent', 'cutoff', 'avg_num_partners_contact_parent', 
        'med_num_partners_contact_parent', 'parent_rank_group', 'avg_num_parents_contact_partner_2', 'med_num_parents_contact_partner_2', 'total_partner_neurons_2',
        'partner_rank_2', 'raw_syn_partner_2', 'parent_reliability_2', 'raw_syn_nern', 'raw_syn_nern_partner_2'
    ]

    partner_weight_input = (df_fin_input
                            .drop(columns=['pcnt_input_to_parent' , *drop_cols])
                            .drop_duplicates(subset=['partner_cell', 'med_partner_pcnt_input'])
                            .rename(columns={'med_raw_syn_nern' : 'med_syn_nern_input', 'med_raw_syn_nern_partner': 'med_syn_nern_partner_input'}))
    partner_weight_output = (df_fin_output
                             .drop(columns=['pcnt_output_from_parent' , *drop_cols])
                             .drop_duplicates(subset=['partner_cell', 'med_partner_pcnt_output'])
                             .rename(columns={'med_raw_syn_nern' : 'med_syn_nern_output', 'med_raw_syn_nern_partner': 'med_syn_nern_partner_output'}))
    
    # # Perform Merge 
    merged_partner_num = partner_num_input.merge(partner_num_output, on='partner_cell', how='outer')
    merged_partner_weight = partner_weight_input.merge( partner_weight_output, on='partner_cell', how='outer')
    merged_partner_df = merged_partner_num.merge(
    merged_partner_weight,
    on='partner_cell',
    how='outer')

    empty_col_num = ['freq_input_all_ranks', 'freq_output_all_ranks', 'med_partner_pcnt_input', 'med_partner_pcnt_output', 'med_syn_nern_output', 'med_syn_nern_input', 'med_syn_nern_partner_output', 'med_syn_nern_partner_input']
    merged_partner_df[empty_col_num] = merged_partner_df[empty_col_num].fillna(0)

    # Create masked color column
    merged_partner_df['color_masked'] = np.where(
        (merged_partner_df['freq_input_all_ranks'] >= 31) |
        (merged_partner_df['freq_output_all_ranks'] >= 31),
        merged_partner_df['med_partner_pcnt_input'],   # keep real values
        np.nan            # everything else becomes NaN
    )
    
    merged_partner_df['color_masked'] = np.where(
        (merged_partner_df['freq_input_all_ranks'] >= 31) |
        (merged_partner_df['freq_output_all_ranks'] >= 31),       
        merged_partner_df['med_partner_pcnt_output'],   # keep real values
        np.nan            # everything else becomes NaN
    )

    return merged_partner_df
def top_partners_freq_in_and_output_plot(merged_partner_df, x_cutoff, y_cutoff, direction_color):
    """ 
    Plot input and output plot 
    - color: color the plot by either input or output strength 
    """


    # Plot
    num_plot = px.scatter(
    merged_partner_df,
    x='freq_input_all_ranks',
    y='freq_output_all_ranks',
    color='color_masked',
    marginal_x='histogram',
    marginal_y='histogram',
    hover_data='partner_cell')
    
    num_plot.update_layout(plot_bgcolor='white', 
                           title=f"Partner Hub Plot (Number of Partners I/O) with Cutoff Thresholds %:{x_cutoff} Reliability:{y_cutoff}. Colored by {direction_color}",
                            xaxis_title='Input',
                            yaxis_title='Output',
                            xaxis2_title=None,
                            yaxis2_title=None)

    return num_plot



# How to define an edge with weights? 
# get partners that give cumulative sum of 95% for EACH parent 
def get_cumulative_sum_95_src(direction, parent_name): 
    """
    Plot cumulative summ till 95% for cell type. All cell types are plotted.
        Arguments:
            
        - direction: This is the direction of the connections.
            
        - parent_name: This is the name of the parent celltype perspective you are looking at.
            - eg LC10a

        NOTE: ensure the arguments are in the same direction! (Input/Output)

        Returns: 
        - df for get input/ output connection
        - result_95: the number of celltype partners it takes to reach 95% cumulative sum df
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    partner_folder = "partner_data_28_04_2026"

    if direction not in ['input', 'output']:
        raise ValueError(f"Direction '{direction}' not recognized")

    if direction == 'input':
        index = 'post_index'
        type_names = 'type_pre'
        filename = f"input_{parent_name}_df.pkl"
        filepath = data_dir / partner_folder / filename
        with open(filepath, "rb") as f:
            df_input = pickle.load(f)
        _, _, _, df = pre_HP.get_input_connections_processing(df_input)


    elif direction == 'output':
        index = 'pre_index'
        type_names = 'type_post'
        filename = f"output_{parent_name}_df.pkl"
        filepath = data_dir / partner_folder / filename
        with open(filepath, "rb") as f:
            df_output = pickle.load(f)
        _, _, _, df = pre_HP.get_output_connections_processing(df_output)

    type_pcnt = 'median_pcnt_ct_total_ct'
    plot_pcnt = 'popul_cumulative_sum'

    # THIS IS NOW A LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or syn information)
    df  = df.drop_duplicates(subset=[type_names, type_pcnt], keep='first')  
    df  = df.sort_values(by=type_pcnt, ascending=False).reset_index(drop=True)
    df.loc[:,index] = df.index + 1 # number of cell types from parent
    df.loc[:,plot_pcnt] = df[type_pcnt].cumsum()  

    # find 95% point 
    threshold = 95
    df_95 = df[df[plot_pcnt] >= threshold]

    if df_95.empty:
        x_95 = None
        y_95 = None
    else:
        x_95 = df_95[index].iloc[0]
        y_95 = df_95[plot_pcnt].iloc[0]

    result_95 = pd.DataFrame({"x_95": [x_95], "y_95": [y_95]})                            
    return df,result_95
def plot_cumulative_sum_95_src(df, direction, parent_name, result_95): 
    """
    Plot cumulative summ till 95% for cell type. All cell types are plotted.
        Arguments:
        - celltype_df: This is the dataframe of all connections to the celltype. e.g: ordered_input, Duplicates are dropped. 
            
        - direction: This is the direction of the connections.
            
        - parent_name: This is the name of the parent celltype perspective you are looking at.
            - eg LC10a
        
        - result_95: the number of celltype partners it takes to reach 95% cumulative sum df

        NOTE: ensure the arguments are in the same direction! (Input/Output)

        Returns: 
        - A line plot for a cell type.
    """

    if direction not in ['input', 'output']:
        raise ValueError(f"Direction '{direction}' not recognized")

    if direction == 'input':
        index = 'post_index'
        type_names = 'type_pre'

    elif direction == 'output':
        index = 'pre_index'
        type_names = 'type_post'

    plot_pcnt = 'popul_cumulative_sum'
    plot_title = parent_name + ': Normalized Cumulative Sum (Population) in direction: ' + direction 

    plot = go.Figure()

    hover_text = [
        f"{type_names}: {row[type_names]}<br>"
        f"{plot_pcnt}: {round(row[plot_pcnt], 2)}<br>"
        f"cellweighted sum: {round(row['median_pcnt_ct_total_ct'], 2)}"
        for _, row in df.iterrows()
    ]
    hover_text = ["start"] + hover_text

    if result_95['x_95'] is not None:
        # Vertical line at x_95
        plot.add_shape(
            type="line",
            x0=result_95['x_95'].iloc[0], x1=result_95['x_95'].iloc[0],
            y0=0, y1=df[plot_pcnt].max(),
            line=dict(color="red", width=2, dash="dash")
        )

    plot.add_trace(go.Scatter(
        x=pd.concat([pd.Series(0), df[index]]), # index of cell types
        y=pd.concat([pd.Series(0), df[plot_pcnt]]), # cumulative sum 
        mode='lines+markers',
        text=hover_text,
        line={"width": 1},
        showlegend=False
    ))

    plot.update_layout(
        title={'text': plot_title},
        xaxis_title="Number of Cell Type Ordered",
        yaxis_title = parent_name + "Cumulative Percent of Population",
        xaxis={"rangemode": "tozero"},
        yaxis={"rangemode": "tozero"},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
                                    
    return plot.show()


# See distributions of ALL partners and cutoff (we didnt find much using this method of defining edges)
def get_all_parent_partner_95(direction, all_celltypes_list_greater_or_eq_10):
    """
    Get all pcnt of partners till 95% cumulative sum (result_95) for all cell type.
    Arguments:
        - direction: This is the direction of the connections.
            
        - all_celltypes_list_greater_or_eq_10: This is a list of parent cell types. 

        NOTE: ensure the arguments are in the same direction! (Input/Output)

    Returns: 
        - A line plot for a cell type.
    """
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])

    if direction not in ['input', 'output']:
        raise ValueError(f"Direction '{direction}' not recognized")
    
    all_parents_partners = []
    for parent_name in all_celltypes_list_greater_or_eq_10:

        if direction == 'input':
            type_names = 'type_pre'
            index = 'post_index'
            drop_list = ['bodyId_post','sum_weight_by_postId', 'sum_typepre_to_postId', 'pcnt_ct_total_ct', 'num_neurons_parent_ct','num_neurons_top_ct', 'frac_parent_neurons_top_ct_seen',
                         'weighted_pcnt_ct_total_ct', 'weight_median_pcnt_ct_total_ct']
            rename_col = {"med_syn_typepre_to_post" : 'raw_syn'}
            filename = "df_all_parents_partners_95_input.pkl"

        elif direction == 'output':
            type_names = 'type_post'
            index = 'pre_index'
            drop_list = ['bodyId_pre','sum_weight_by_preId', 'sum_typepost_from_preId', 'pcnt_ct_total_ct', 'num_neurons_parent_ct','num_neurons_top_ct', 'frac_parent_neurons_top_ct_seen',
                'weighted_pcnt_ct_total_ct', 'weight_median_pcnt_ct_total_ct']
            rename_col = {"med_syn_typepost_from_pre" : 'raw_syn'}
            filename = "df_all_parents_partners_95_output.pkl"

        df, result_95 = get_cumulative_sum_95_src(direction, parent_name)  # THIS IS A LVL 3 df: one row per type (pre/post) per parent cell (median). NO individual Parent cell information or syn information)
        df = df.drop(columns= drop_list)
        df = df.rename(columns=rename_col)
        
        if result_95 is None:
            df_cut95 = df 
        else:
            df_cut95 = df[df[index] < result_95['x_95'].loc[0]]
        all_parents_partners.append(df_cut95)

    df_all_parents_partners = pd.concat(all_parents_partners, ignore_index=True)

    # Dump in directory 
    with open(data_dir / filename, "wb") as f:
        pickle.dump(df_all_parents_partners, f)
    return df_all_parents_partners
def plot_all_partner_distribution_95(direction, x_axis, bins, log):
    """
    Plot distribution pcnt of partners till 95% cumulative sum (result_95) for cell type.
    Arguments:

    - direction: This is the direction of the connections.
    - x_axis: what to plot: raw_syn or median_pcnt_ct_total_ct
    - log = log x axis (Y/N) 
    - bins: number fo bins to add (add number: eg: 100)

    NOTE: ensure the arguments are in the same direction! (Input/Output)

    Returns: 
        - A line plot for a cell type.
        - A df_all_parents_partners_95
    """
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    if direction not in ['input', 'output']:
        raise ValueError(f"Direction '{direction}' not recognized")
    
    # Load once (not inside loop)
    if direction == 'input':
        type_names = 'type_post'
        filepath = data_dir / "df_all_parents_partners_95_input.pkl"
    else:
        type_names = 'type_pre'
        filepath = data_dir / "df_all_parents_partners_95_output.pkl"

    with open(filepath, "rb") as f:
        df_all_parents_partners_95 = pickle.load(f)

    df_all_parents_partners_95["log_x"] = np.log10(df_all_parents_partners_95[x_axis])

    if log == 'Y':
        fig = px.histogram(
        df_all_parents_partners_95,
        x="log_x",
        nbins=bins,
        opacity=0.6,
        # color=type_names,
        range_x=[df_all_parents_partners_95["log_x"].min(), df_all_parents_partners_95["log_x"].max()]
    )

        fig.update_layout(
            title="Partner Distribution (90% cumulative) Across Parent Cell Types in direction: " + direction + "bins: " + str(bins),
            xaxis_title="log_x: " + x_axis,
            yaxis_title="Count",
            yaxis_type="log",
            barmode='stack'
        )
    
    if log == 'N':
        fig = px.histogram(
        df_all_parents_partners_95,
        x=x_axis,
        nbins=bins,
        opacity=0.6,
        # color=type_names,
        range_x=[df_all_parents_partners_95[x_axis].min(), df_all_parents_partners_95[x_axis].max()]
    )

        fig.update_layout(
            title="Partner Distribution (90% cumulative) Across Parent Cell Types in direction: " + direction + "bins: " + str(bins),
            xaxis_title=x_axis,
            yaxis_title="Count",
            yaxis_type="log",
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white'
            )

        fig.update_yaxes(type="log" , title_font=dict(size=20), tickfont=dict(size=10))
        fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=10))

    fig.show()

    return df_all_parents_partners_95

# Hub Matrix and CLustering 
def parent_hub_df(merged_partner_df, all_celltypes_list_greater_or_eq_10, data_type, threshold_in=None, threshold_out=None):
    hub_neurons = merged_partner_df[merged_partner_df['color_masked'].notna()]['partner_cell'].to_list()

    if data_type == 'topn':
        df_fin_out, _, _  = top_ct_partner_freq_df(3, 'output')
        df_fin_in, _, _  = top_ct_partner_freq_df(3, 'input')

    if data_type == 'reliability':
        df_fin_in, _, _ = top_ct_partner_freq_reliability_df(all_celltypes_list_greater_or_eq_10, 'Y', 'input', 3, 0.8)
        df_fin_out, _, _ = top_ct_partner_freq_reliability_df(all_celltypes_list_greater_or_eq_10, 'Y', 'output', 3, 0.8)

    hub_in_df = df_fin_in[df_fin_in['partner_cell'].isin(hub_neurons)]
    hub_out_df = df_fin_out[df_fin_out['partner_cell'].isin(hub_neurons)]

    drop_cols = [
        'raw_syn', 'num_neurons_parent_ct', 'partner_reliability', 'cutoff', 'avg_num_partners_contact_parent', 'med_num_partners_contact_parent', 'med_raw_syn_nern']
    hub_in_df = hub_in_df.drop(columns=['total_partner_pcnt_input', 'med_partner_pcnt_input', *drop_cols])
    hub_out_df = hub_out_df.drop(columns=['total_partner_pcnt_output', 'med_partner_pcnt_output', *drop_cols])

    # Matrix processing 
    hub_in_heatmap = hub_in_df.pivot(
    index='parent',
    columns='partner_cell',
    values='pcnt_input_to_parent'
    ).fillna(0)

    hub_out_heatmap = hub_out_df.pivot(
        index='parent',
        columns='partner_cell',
        values='pcnt_output_from_parent'
    ).fillna(0)

    # if threshold is yes
    if threshold_in is not None and threshold_out is not None:

        hub_in_heatmap = hub_in_heatmap.loc[(hub_in_heatmap > threshold_in).any(axis=1),(hub_in_heatmap  > threshold_in).any(axis=0)]
        hub_out_heatmap = hub_out_heatmap.loc[(hub_out_heatmap > threshold_out).any(axis=1),(hub_out_heatmap > threshold_out).any(axis=0)]

    all_parents = sorted(set(hub_in_heatmap.index) | set(hub_out_heatmap.index))
    all_partners = sorted(set(hub_in_heatmap.columns) | set(hub_out_heatmap.columns))

    hub_in_full = hub_in_heatmap.reindex(index=all_parents, columns=all_partners).fillna(0)
    hub_out_full = hub_out_heatmap.reindex(index=all_parents, columns=all_partners).fillna(0)

    hub_in_full = hub_in_full + 1e-9
    hub_out_full = hub_out_full + 1e-9

    return hub_in_full,  hub_out_full
def parent_hub_plot_scatter(merged_partner_df, all_celltypes_list_greater_or_eq_10, data_type):
    # Query
    hub_neurons = merged_partner_df[merged_partner_df['color_masked'].notna()]['partner_cell'].to_list()
    if data_type == 'reliability':
        hub_out_df, hub_in_df, _ , _ = parent_hub_df(merged_partner_df,  all_celltypes_list_greater_or_eq_10, 'reliability')
    if data_type == 'topn':
        hub_out_df, hub_in_df, _ , _ = parent_hub_df(merged_partner_df,  all_celltypes_list_greater_or_eq_10, 'topn')
    
    colors = px.colors.qualitative.Plotly
    hub_colors = {}
    for i, hub in enumerate(hub_neurons):
        color_index = i % len(colors)
        hub_colors[hub] = colors[color_index]

    hub_parent_plot = make_subplots(rows=2, cols=1,
                                    shared_xaxes=True,
                                    subplot_titles=["Hubs Recieve Output from Parents", "Hubs Send Input to Parents"])

    for hub in hub_neurons:
        output = hub_out_df[hub_out_df['partner_cell'] == hub]
        input = hub_in_df[hub_in_df['partner_cell'] == hub]
        hub_parent_plot.add_trace(
            go.Scatter(
                x=output['parent'], 
                y=output['pcnt_output_from_parent'],
                text=output['partner_cell'],
                customdata=output['pcnt_output_from_parent'],
                hovertemplate=
                    "<b>Hub:</b> %{text}<br>" +
                    "<b>Parent:</b> %{x}<br>" +
                    "<b>Output Strength:</b> %{customdata}<extra></extra>",
                mode='markers',
                name=hub,
                legendgroup=hub,
                marker=dict(
                    size=10,
                    color=hub_colors[hub]
                ),
                showlegend=True),
                row=1, col=1)
        hub_parent_plot.add_trace(
            go.Scatter(
                x=input['parent'], 
                y=input['pcnt_input_to_parent'],
                text=input['partner_cell'],
                customdata=input['pcnt_input_to_parent'],
                hovertemplate=
                    "<b>Hub:</b> %{text}<br>" +
                    "<b>Parent:</b> %{x}<br>" +
                    "<b>Input Strength:</b> %{customdata}<extra></extra>",
                mode='markers',
                name=hub,
                legendgroup=hub,
                marker=dict(
                    size=10,
                    color=hub_colors[hub]
                ),
                showlegend=False),
                row=2, col=1)
        
    hub_parent_plot.update_xaxes(
        title_text="Parent Cell Type",
        row=2, col=1
    )

    hub_parent_plot.update_yaxes(
        title_text="Percent Output From Parent",
        row=1, col=1
    )

    hub_parent_plot.update_yaxes(
        title_text="Percent Input To Parent",
        row=2, col=1
    )
    hub_parent_plot.show()
def clustering_hubs_parents_I_O(hub_in_full, hub_out_full):
    df_hub_comb = pd.concat([hub_in_full, hub_out_full], axis=0, join='outer')
    # df_parent_comb = pd.merge(hub_in_full, hub_out_full, left_index=True, right_index=True, how='outer')

    # Column clustering:
    # Clusters hubs based on their connectivity to the focal cell types.
    hub_dist  = pdist(df_hub_comb.T, metric='cosine')
    hub_linkage = linkage(hub_dist, method='average')
    # Get dendrogram leaf orders
    hub_order = leaves_list(hub_linkage)

    # 2. INPUT FOCAL CLUSTERING: Focals are clustered based only on their input profiles from the hubs.
    focal_in_dist = pdist(hub_in_full, metric='cosine')
    focal_in_linkage = linkage(focal_in_dist, method='average')
    # Get dendrogram leaf orders
    focal_in_order = leaves_list(focal_in_linkage)

    # 3. OUTPUT FOCAL CLUSTERING: Focals are clustered based only on their output profiles from the hubs.
    focal_out_dist = pdist(hub_out_full, metric='cosine')
    focal_out_linkage = linkage(focal_out_dist, method='average')
    # Get dendrogram leaf orders
    focal_out_order = leaves_list(focal_out_linkage)

    # Reorder matrices
    sorted_in = hub_in_full.iloc[focal_in_order, hub_order]
    sorted_out = hub_out_full.iloc[focal_out_order, hub_order]

    # Create figure layout
    fig = plt.figure(figsize=(40, 30))

    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=[2, 40],
        height_ratios=[5, 20, 20, 5],
        wspace=0.02,
        hspace=0.2
    )


    # hub input dendrogram
    hub_in_dendro = fig.add_subplot(gs[1, 0])
    dendrogram(
        hub_linkage,
        ax=hub_in_dendro,
        orientation='left',
        no_labels=True,
        color_threshold=None
    )
    hub_in_dendro.axis('off')

    # INPUT FOCAL DENDROGRAM
    focal_in_dendro = fig.add_subplot(gs[0, 1])
    dendrogram(
        focal_in_linkage,
        ax=focal_in_dendro,
        orientation='top',
        no_labels=True,
        color_threshold=None
    )

    focal_in_dendro .axis('off')

    # INPUT HEATMAP

    ax_in = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        sorted_in.T,
        cmap='Blues',
        vmin=0,
        vmax=60,
        ax=ax_in,
        cbar=False
    )

    ax_in.set_ylabel("Hub Neurons (Senders)", fontsize=18)
    ax_in.set_xlabel(
        "Focal Cell Types Receiving Input Connections from Hubs",
            fontsize=18
    )
    ax_in.xaxis.set_label_position('top')

    # OUTPUT FOCAL DENDROGRAM
    focal_out_dendro = fig.add_subplot(gs[3, 1])
    dendrogram(
        focal_out_linkage,
        ax=focal_out_dendro,
        orientation='bottom',
        no_labels=True,
        color_threshold=None
    )
    focal_out_dendro.axis('off')

    
    #Output Heatmap 
    ax_out = fig.add_subplot(gs[2, 1])

    sns.heatmap(
        sorted_out.T,
        cmap='Blues',
        vmin=0,
        vmax=40,
        ax=ax_out,
        cbar=False
    )

    ax_out.set_ylabel("Hub Neurons (Receivers)", fontsize=18)
    ax_out.set_xlabel(
        "Focal Cell Types Sending Output Connections to Hubs",
            fontsize=18
    )
    ax_out.xaxis.set_label_position('top')

    # Format heatmap axes
    for ax in [ax_in, ax_out]:

        ax.tick_params(
            axis='x',
            rotation=90,
            labelsize=10
        )

        ax.tick_params(
            axis='y',
            rotation=0,
            labelsize=10
        )


    plt.tight_layout()

    return fig, sorted_in, sorted_out

def hub_bipartites_only_df(merged_partner_df,cut_bipartite_parent_input_v3, cut_bipartite_parent_output_v3):
    #  HUB Neurons -- run this to generate bipartite plots
    hub_neurons = merged_partner_df[merged_partner_df['color_masked'].notna()]['partner_cell'].to_list()
    bipartite_parent_input_v3_hubs = cut_bipartite_parent_input_v3[
        cut_bipartite_parent_input_v3['parent'].isin(hub_neurons) &
        cut_bipartite_parent_input_v3['partner_cell'].isin(hub_neurons)
    ]
    bipartite_parent_output_v3_hubs = cut_bipartite_parent_output_v3[
        cut_bipartite_parent_output_v3['parent'].isin(hub_neurons) &
        cut_bipartite_parent_output_v3['partner_cell'].isin(hub_neurons)
    ]
    return bipartite_parent_input_v3_hubs , bipartite_parent_output_v3_hubs, hub_neurons
