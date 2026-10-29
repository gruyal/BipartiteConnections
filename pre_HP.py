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
import inspect
import yaml
from pathlib import Path
import pickle


def configpath(config_path='/Users/aarushivardhan/Documents/connectome/msc_av/config.yaml'): 
    """
    Load the config.yaml file and return a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration data from the YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# (1) Fetch data only (no processing) NOTE: these cell types need to be nto queried issue with red exp {'MeVP6_unclear', None, 'Pm7_Li28', 'T5a_unclear'}
def get_input_connections(input_celltype, side): 
    """
    Fetch all input connections going into cell type.
    Arguments:
    - input_to_celltype:  cell type. Give only one cell type for now, eg: LC10a.
        - Keep note: (LC10[a-d]|LC10c-.*).
    - side: neuron on left or right? E.g CT1 cell body is on the left side (even thoa right side OL neuron) 

    Returns:
    1. og_input_to_celltype_df: df of all input connections to celltype. Drop duplicates done. 
    """ 
    # (lvl 0) Pre-process and query data 
    if side == 'right':
        exp=f"^{input_celltype}(?![0-9A-Za-z]).*_R$"

    if side == 'left':
        exp=f"^{input_celltype}(?![0-9A-Za-z]).*_L$"

    if side == 'middle': 
        exp=f"^{input_celltype}(?![0-9A-Za-z]).*_M$"

    
    neuron_in_df, conn_in_df = fetch_adjacencies(targets = NC(instance=exp, regex= True, status='Traced'), sources = None)
    og_input_to_celltype_df = pd.merge(neuron_in_df, conn_in_df, left_on='bodyId', right_on='bodyId_pre')
    og_input_to_celltype_df = pd.merge(neuron_in_df, og_input_to_celltype_df, left_on='bodyId', right_on='bodyId_post')
    og_input_to_celltype_df = og_input_to_celltype_df.drop(['bodyId_x', 'bodyId_y','instance_x','instance_y'], axis=1)
    og_input_to_celltype_df  = og_input_to_celltype_df.rename(columns={'type_x': 'type_post'})
    og_input_to_celltype_df  = og_input_to_celltype_df.rename(columns={'type_y': 'type_pre'})
    og_input_to_celltype_df = og_input_to_celltype_df[og_input_to_celltype_df['roi'] != 'NotPrimary']

    # (lvl 1) remove duplicates
    og_input_to_celltype_df = og_input_to_celltype_df.groupby(['bodyId_pre', 'bodyId_post']).agg({ 
            'weight': 'sum',  
            'type_pre': 'first',
            'type_post': 'first',
            'roi': 'first'
        }).reset_index()
    og_input_to_celltype_df.sort_values(by='weight', ascending=True)     
    og_input_to_celltype_df = og_input_to_celltype_df.dropna(subset=['type_pre'])

    return og_input_to_celltype_df
def get_output_connections(output_celltype, side): 
    """
    Fetch all input connections going into cell type.
    Arguments:
    - output_to_celltype:  cell type. Give only one cell type for now, eg: LC10a.
        - Keep note: (LC10[a-d]|LC10c-.*).
    - side: neuron on left or right? E.g CT1 cell body is on the left side (even thoa right side OL neuron) 

    Returns:
    1. og_output_to_celltype_df: df of all output connections to celltype. Drop duplicates done. 
    """ 
    # (lvl 0) Pre-process and query data 
    if side == 'right':
        exp=f"^{output_celltype}(?![0-9A-Za-z]).*_R$"

    if side == 'left':
        exp=f"^{output_celltype}(?![0-9A-Za-z]).*_L$"

    if side == 'middle': 
        exp=f"^{output_celltype}(?![0-9A-Za-z]).*_M$"

    # (lvl 0) Pre-process and query data 
    neuron_out_df, conn_out_df = fetch_adjacencies(sources = NC(instance=exp, regex= True, status='Traced'), targets = None)
    og_output_to_celltype_df = pd.merge(neuron_out_df, conn_out_df, left_on='bodyId', right_on='bodyId_pre')
    og_output_to_celltype_df = pd.merge(neuron_out_df, og_output_to_celltype_df, left_on='bodyId', right_on='bodyId_post')
    og_output_to_celltype_df = og_output_to_celltype_df.drop(columns=['bodyId_x', 'bodyId_y','instance_x','instance_y'])
    og_output_to_celltype_df = og_output_to_celltype_df.rename(columns={'type_x': 'type_post'})
    og_output_to_celltype_df  = og_output_to_celltype_df.rename(columns={'type_y': 'type_pre'}) 
    og_output_to_celltype_df = og_output_to_celltype_df[og_output_to_celltype_df['roi'] != 'NotPrimary']

    # (lvl 1) remove duplicates
    og_output_to_celltype_df = og_output_to_celltype_df.groupby(['bodyId_pre', 'bodyId_post']).agg({
                    'weight': 'sum',  
                    'type_pre': 'first',
                    'type_post': 'first',
                    'roi':'first'
                }).reset_index()
    og_output_to_celltype_df.sort_values(by='weight', ascending=True)     
    og_output_to_celltype_df = og_output_to_celltype_df.dropna(subset=['type_post'])

    return og_output_to_celltype_df

# (2) This is the processing part (calculations)
def get_input_connections_processing(og_input_to_celltype_df, num_of_top_inputs=None):
    """
    Process all input connections going into cell type.
    Arguments:
    - og_input_to_celltype_df: df from get_input_connections.
    - num_of_top_inputs: (int or None) Number of top inputs to capture eg: 20. 
                         If None, returns all inputs (no filtering).

    Returns:
    An individual data frame contains all input connections to the selected type, with:
    bodyId_pre&post, weight, type_pre&post
    1. ordered_input: (list) all input cell types ordered by median largest -> smallest.
    2. top_input: (list) ordered, input cell type, ordered by median size. With selected number of celltypes.
    3. input_to_celltype_df: df of all input connections to celltype. Drop duplicates done. 
    """
    # Computations:
    input_to_celltype_df = og_input_to_celltype_df.copy()
    # Parent Cell Total Synaptic Weight. (One value per ID)
    input_to_celltype_df['sum_weight_by_postId'] = input_to_celltype_df.groupby('bodyId_post')['weight'].transform('sum')

    # (lvl 1) Individual Pre-Cell / Parent cell
    input_to_celltype_df['pcnt_indv_postsyn_total_postsyn'] = (input_to_celltype_df['weight'] / input_to_celltype_df['sum_weight_by_postId']) * 100

    # (lvl 2) Summed CT syn weight --> Parent ID 
    input_to_celltype_df['sum_typepre_to_postId'] = input_to_celltype_df.groupby(['bodyId_post', 'type_pre'])['weight'].transform('sum')

    # (lvl 1) How many type_pre body ID contact --> 1 Parent ID (paired with above calculation)
    input_to_celltype_df['num_typepres_to_postId'] = input_to_celltype_df.groupby(['bodyId_post', 'type_pre'])['bodyId_pre'].transform('nunique')


    # # (lvl 1) Synapses per type post divided by number of connections to parent
    # input_to_celltype_df['syn_typepres_to_postId_by_conn'] = input_to_celltype_df['sum_typepre_to_postId'] / input_to_celltype_df['num_typepres_to_postId']
    
    # (lvl 2): Type-Pre / Parent ID
    input_to_celltype_df['pcnt_ct_total_ct'] = (input_to_celltype_df['sum_typepre_to_postId'] / input_to_celltype_df['sum_weight_by_postId']) * 100

    # (lvl 3): Median Synapses between a Parent and Partner 
    input_to_celltype_df['med_syn_typepre_to_post'] = input_to_celltype_df.groupby('type_pre')['weight'].transform('median')
    input_to_celltype_df['std_syn_typepre_to_post'] = input_to_celltype_df.groupby('type_pre')['weight'].transform('std')
    input_to_celltype_df['range_syn_typepre_to_post'] = input_to_celltype_df.groupby('type_pre')['weight'].transform('max') - input_to_celltype_df.groupby('type_pre')['weight'].transform('min')


    # NOTE: (lvl 2) Dataframe. Unique pre-ID post-ID combnations are REMOVED. NOW: Perspective of Parent cell (ID) and its Cell Types (type-pre).
    input_to_celltype_df = input_to_celltype_df.drop_duplicates(subset=['bodyId_post', 'type_pre'], keep='first')
    input_to_celltype_df = input_to_celltype_df.drop(columns=['bodyId_pre'])

    # Number of cells (IDs) in each parent (one value). 
    input_to_celltype_df['num_neurons_parent_ct'] = input_to_celltype_df['bodyId_post'].nunique()
    # Number of Parent cells a Pre-Cell type is seen in. (One value per Pre-Cell Type) lvl 3
    input_to_celltype_df['num_neurons_top_ct'] = input_to_celltype_df.groupby('type_pre')['bodyId_post'].transform('nunique')
    # (lvl 3) Parent cells in Pre-CT / Total Parent Cells 
    input_to_celltype_df['frac_parent_neurons_top_ct_seen'] = (input_to_celltype_df['num_neurons_top_ct'] / input_to_celltype_df['num_neurons_parent_ct'])

    # (lvl 2) on average how many type_pre contact --> parent ID 
    input_to_celltype_df['avg_num_typepres_to_postId'] = (input_to_celltype_df.groupby('type_pre')['num_typepres_to_postId'].transform('sum') / input_to_celltype_df['num_neurons_parent_ct'])
    input_to_celltype_df['num_neurons_top_ct_missing'] = (input_to_celltype_df['num_neurons_parent_ct']- input_to_celltype_df['num_neurons_top_ct'])
    def median_with_zeros(group):
        observed = group['num_typepres_to_postId'].values
        n_zero = group['num_neurons_top_ct_missing'].iloc[0]

        full = np.concatenate([observed, np.zeros(int(n_zero))])
        return np.median(full)

    input_to_celltype_df['med_num_typepres_to_postId'] = (
        input_to_celltype_df['type_pre']
        .map(
            input_to_celltype_df
            .groupby('type_pre')
            .apply(median_with_zeros, include_groups=False)
        )
    )
    
    # Weighted pcnt_ct_total_ct
    input_to_celltype_df['weighted_pcnt_ct_total_ct'] = (input_to_celltype_df['pcnt_ct_total_ct'] * input_to_celltype_df['frac_parent_neurons_top_ct_seen'])

    # Median of pcnt_ct_total_ct (One value per Pre-Cell Type)
    input_to_celltype_df['median_pcnt_ct_total_ct'] = input_to_celltype_df.groupby('type_pre')['pcnt_ct_total_ct'].transform('median')


    # Weighted Median of pcnt_ct_total_ct
    input_to_celltype_df['weight_median_pcnt_ct_total_ct'] = (input_to_celltype_df['median_pcnt_ct_total_ct'] * input_to_celltype_df['frac_parent_neurons_top_ct_seen'])

    # # (lvl 3) Median summed CT syn weight --> Parent by connections to partner (median across all parent IDs) 
    # input_to_celltype_df['median_typepre_syn_to_parent'] = input_to_celltype_df.groupby('type_pre')['syn_typepres_to_postId_by_conn'].transform('median')

    # Compute median percent by post type once
    median_by_pre = (
        input_to_celltype_df
        .groupby('type_pre')['median_pcnt_ct_total_ct']
        .median()
        .sort_values(ascending=False)
    )

    ordered_input = median_by_pre.index.tolist()

    # Handle optional argument
    if num_of_top_inputs is not None:
        top_input = median_by_pre.head(num_of_top_inputs).index.tolist()
        weighted_input = (
            input_to_celltype_df
            .groupby('type_pre')['weight_median_pcnt_ct_total_ct']
            .median() #median here to allow sort_values() on SeriesGroupby. Taking a median of values already median'ed' shouldn't matter cause it will stay the same. Can sort. 
            .sort_values(ascending=False)
            .nlargest(num_of_top_inputs)
            .index.tolist()
        )
    else:
        # return full lists if not limiting to top N
        top_input = ordered_input
        weighted_input = (
            input_to_celltype_df
            .groupby('type_pre')['weight_median_pcnt_ct_total_ct']
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    return ordered_input, top_input, weighted_input, input_to_celltype_df
def get_output_connections_processing(og_output_to_celltype_df, num_of_top_outputs=None):
    """
    Fetch all output connections going out of cell type.
    Arguments:
    - og_output_to_celltype_df: df from get_output_connections. 
    - num_of_top_outputs: number of top outputs to capture, ordered by median size. e.g: 20.

    Returns:
    1. ordered_output: all cell type ordered by median size. 
    2. top_output: top cell type based on num_of_top_outputs argument. 
    3. output_to_celltype_df: all output connections with duplicates dropped
    """
    # Computations:
    output_to_celltype_df = og_output_to_celltype_df.copy()
    # Parent Cell Total Synaptic Weight. (One value per ID)
    output_to_celltype_df['sum_weight_by_preId'] = output_to_celltype_df.groupby('bodyId_pre')['weight'].transform('sum')

    # (lvl 1) Individual Post-Cell / Parent cell
    output_to_celltype_df['pcnt_indv_presyn_total_presyn'] = (output_to_celltype_df['weight'] / output_to_celltype_df['sum_weight_by_preId']) * 100

    # (lvl 2) Summed CT syn weight --> Parent ID    
    output_to_celltype_df['sum_typepost_from_preId'] = output_to_celltype_df.groupby(['bodyId_pre', 'type_post'])['weight'].transform('sum')

    # (lvl 1) How many type_pre contact --> 1 Parent ID (paired with above calculation)
    output_to_celltype_df['num_typeposts_from_preId'] = output_to_celltype_df.groupby(['bodyId_pre', 'type_post'])['bodyId_post'].transform('nunique')

    # # (lvl 1) Synapses per type post divided by number of connections to parent
    # output_to_celltype_df['syn_typeposts_from_preId_by_conn'] = output_to_celltype_df['sum_typepost_from_preId'] / output_to_celltype_df['num_typeposts_from_preId']
    
    # (lvl 2): Type-Post / Parent ID
    output_to_celltype_df['pcnt_ct_total_ct'] = (output_to_celltype_df['sum_typepost_from_preId'] / output_to_celltype_df['sum_weight_by_preId']) * 100

    # (lvl 3): Median Synapses between a Parent and Partner 
    output_to_celltype_df['med_syn_typepost_from_pre'] = output_to_celltype_df.groupby('type_post')['weight'].transform('median')
    output_to_celltype_df['std_syn_typepost_from_pre'] = output_to_celltype_df.groupby('type_post')['weight'].transform('std')
    output_to_celltype_df['range_syn_typepost_from_pre'] = output_to_celltype_df.groupby('type_post')['weight'].transform('max') - output_to_celltype_df.groupby('type_post')['weight'].transform('min')



    # NOTE: (lvl 2) Dataframe. Unique pre-ID post-ID combnations are REMOVED. NOW: Perspective of Parent cell (ID) and its Cell Types (type-post).
    output_to_celltype_df = output_to_celltype_df.drop_duplicates(subset=['bodyId_pre', 'type_post'], keep='first')
    output_to_celltype_df = output_to_celltype_df.drop(columns=['bodyId_post'])

    # Number of cells (IDs) in each parent (one value). 
    output_to_celltype_df['num_neurons_parent_ct'] = output_to_celltype_df['bodyId_pre'].nunique()
    # Number of Parent cells a Post-Cell type is seen in. (One value per Post-Cell Type) lvl 3 
    output_to_celltype_df['num_neurons_top_ct'] = output_to_celltype_df.groupby('type_post')['bodyId_pre'].transform('nunique')
    # (lvl 3) Parent cells in Post-CT / Total Parent Cells 
    output_to_celltype_df['frac_parent_neurons_top_ct_seen'] = (output_to_celltype_df['num_neurons_top_ct'] / output_to_celltype_df['num_neurons_parent_ct'])


    # (lvl 2) on average how many type_post contact --> parent ID 
    output_to_celltype_df['avg_num_typeposts_to_preId'] = (output_to_celltype_df.groupby('type_post')['num_typeposts_from_preId'].transform('sum') / output_to_celltype_df['num_neurons_parent_ct'])
    output_to_celltype_df['num_neurons_top_ct_missing'] = (output_to_celltype_df['num_neurons_parent_ct']- output_to_celltype_df['num_neurons_top_ct'])
    def median_with_zeros(group):
        observed = group['num_typeposts_from_preId'].values
        n_zero = group['num_neurons_top_ct_missing'].iloc[0]

        full = np.concatenate([observed, np.zeros(int(n_zero))])
        return np.median(full)

    output_to_celltype_df['med_num_typeposts_to_preId'] = (
        output_to_celltype_df['type_post']
        .map(
            output_to_celltype_df
            .groupby('type_post')
            .apply(median_with_zeros, include_groups=False)
        )
    )

    # Weighted pcnt_ct_total_ct
    output_to_celltype_df['weighted_pcnt_ct_total_ct'] = (output_to_celltype_df['pcnt_ct_total_ct'] * output_to_celltype_df['frac_parent_neurons_top_ct_seen'])

    # Median of pcnt_ct_total_ct (One value per Post-Cell Type) 
    output_to_celltype_df['median_pcnt_ct_total_ct'] = output_to_celltype_df.groupby('type_post')['pcnt_ct_total_ct'].transform('median')

    # Weighted Median of pcnt_ct_total_ct
    output_to_celltype_df['weight_median_pcnt_ct_total_ct'] = (output_to_celltype_df['median_pcnt_ct_total_ct'] * output_to_celltype_df['frac_parent_neurons_top_ct_seen'])

    # # (lvl 3) Parent --> median summed CT syn weight by connections to partner (median across all parent IDs) 
    # output_to_celltype_df['median_typepost_syn_from_parent'] = output_to_celltype_df.groupby('type_post')['syn_typeposts_from_preId_by_conn'].transform('median')

    # Compute median percent by post type once
    median_by_post = (
        output_to_celltype_df
        .groupby('type_post')['median_pcnt_ct_total_ct']
        .median()
        .sort_values(ascending=False)
    )
    ordered_output = median_by_post.index.tolist()

    # Handle optional argument
    if num_of_top_outputs is not None:
        top_output = median_by_post.head(num_of_top_outputs).index.tolist()
        weighted_output = (
            output_to_celltype_df
            .groupby('type_post')['weight_median_pcnt_ct_total_ct']
            .median() #median here to allow sort_values() on SeriesGroupby. Taking a median of values already median'ed' shouldn't matter cause it will stay the same. Can sort. 
            .sort_values(ascending=False)
            .nlargest(num_of_top_outputs)
            .index.tolist()
        )
    else:
        # return full lists if not limiting to top N
        top_output = ordered_output
        weighted_output = (
            output_to_celltype_df
            .groupby('type_post')['weight_median_pcnt_ct_total_ct']
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    return ordered_output, top_output, weighted_output, output_to_celltype_df


# Plot top 20 inputs and all inputs ordered by median size
def plot_connections(plot_type, ct_list, celltype_df, direction, parent_name):
    """
    Plot the connections output/input from a Parent cell type (cell type (CT) chosen).
    Arguments:
    - plot_type: 'all' (all ct_list given), 'top' (top ct_list given), 'weighted' (weighted ct_list)

    - ct_list: ordered_input, top_input, weighted_input,

    - celltype_df: This is the dataframe of all connections to the Parent celltype. 
        ('output_to_celltype_df' / 'input_to_celltype_df')

    - direction: This is the direction of the connections. 
        ('input' or 'output')

    - parent_name: This is the name of the celltype (or Parent) you are looking at.
        - This is important for the title of the plot.
        - e.g: LC10a, TmY, etc.

    NOTE: ensure the arguments are in the same direction! (Input/Output)

    Returns: 
    - fig of one of three types
    """

    if direction == 'input':
        x='type_pre'
        hover_data=['bodyId_post']
        color_discrete_sequence=['green']
        labels={
            x: f'Pre-cell type to {parent_name}',
            'pcnt_ct_total_ct': f'Pre-cell → {parent_name} (% of Total Connections)',
            'weighted_pcnt_ct_total_ct': f'Pre-cell → {parent_name} (Weighted %)'
        }
        
    elif direction == 'output': 
            x='type_post'
            hover_data=['bodyId_pre'] 
            color_discrete_sequence=['blue']
            labels={
                x: f'Post-cell type from {parent_name}',
                'pcnt_ct_total_ct': f'{parent_name} → Post-Cell (% of Total Connections)',
                'weighted_pcnt_ct_total_ct': f'{parent_name} → Post-Cell (Weighted %)'
            }
    else:
            raise ValueError(f"Direction '{direction}' not recognized. Use 'input' or 'output'.")
    
    category_orders = {x: ct_list} if plot_type in ['top', 'weighted'] else None

    if plot_type == 'all':
        df_plot = celltype_df
        y='pcnt_ct_total_ct'
        title = f"Percent Distribution of Cell Types from Parent: {parent_name} in direction: {direction}. Plot type: {plot_type}"
    elif plot_type in ['top', 'weighted']:
        df_plot = celltype_df[celltype_df[x].isin(ct_list)]
        title_type = 'Top' if plot_type == 'top' else 'Top Weighted'
        y ='weighted_pcnt_ct_total_ct' if plot_type == 'weighted' else 'pcnt_ct_total_ct'
        title = f"Percent Distribution of {title_type} {len(ct_list)} Cell Types from Parent: {parent_name} in direction: {direction}. Plot type: {plot_type}"
    else:
        raise ValueError(f"Plot type '{plot_type}' not recognized")

    fig  = px.box(
            df_plot,
            x=x,
            y=y,
            hover_data= hover_data, 
            points=False,
            color_discrete_sequence=color_discrete_sequence,
            labels= labels, 
            category_orders=category_orders , 
            title=title
        )
       
    # Make background white and remove gridlines
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# This is for synapses NOT pcnt (see above). Single cell level, save for later
def plot_type_syn(ct_list, celltype_df, direction, parent_name):
    """
    Plot num of syn from a Parent cell type (cell type (CT) chosen) to output/input .
    Arguments:
    - ct_list: ordered_input, top_input, weighted_input,

    - celltype_df: This is the dataframe of all connections to the Parent celltype. 
        ('output_to_celltype_df' / 'input_to_celltype_df')

    - direction: This is the direction of the connections. 
        ('input' or 'output')

    - parent_name: This is the name of the celltype (or Parent) you are looking at.
        - This is important for the title of the plot.
        - e.g: LC10a, TmY, etc.

    NOTE: ensure the arguments are in the same direction! (Input/Output)

    Returns: 
    - fig 
    """
    if direction == 'input':
        x='type_pre'
        y='syn_typepres_to_postId_by_conn'
        hover_data=['bodyId_post']
        color_discrete_sequence=['green']
        labels={
            x: f'Pre-cell type to {parent_name}',
            y: f'Pre-cell → {parent_name} (number of synapses shared between cell types divided by connections)'
        }
        
    elif direction == 'output': 
        x='type_post'
        y='syn_typeposts_from_preId_by_conn'
        hover_data=['bodyId_pre'] 
        color_discrete_sequence=['blue']
        labels={
            'type_post': 'Post-cell type to' + parent_name,
            'syn_typeposts_from_preId_by_conn': parent_name +  '(number of synapses shared between cell types divided by connections)'
        }
    else:
        raise ValueError(f"Direction '{direction}' not recognized. Use 'input' or 'output'.")
    
    category_orders={x: ct_list}

    plot_type = 'ct_syn'
    df_plot = celltype_df
    title = f"Number of Synapses between cell types from Parent divded by connections: {parent_name} in direction: {direction}. Plot type: {plot_type}"

    fig  = px.box(
        df_plot,
        x=x,
        y=y,
        hover_data= hover_data, 
        points=False,
        color_discrete_sequence=color_discrete_sequence,
        labels= labels, 
        category_orders=category_orders , 
        title=title
        )
       
    return fig

# Stacked bar plot 
def plot_stacked(all_celltype, num_celltype_in_plot, celltype_df, direction, parent_name):
    """
        Plot the a stacked bar plot for top input or output cell from chosen cell type.
        Arguments:
        - all_celltype: a list of inputs/outputs. e.g: ordered_input
            
        - num_celltype_in_plot: Indicate # of celltypes to plot.

        - celltype_df: This is the dataframe of all connections to the celltype. 
            ('output_to_celltype_df' / 'input_to_celltype_df')
           
        - direction: This is the direction of the connections. ('input' or 'output')

        - parent_name: This is the name of the parent celltype perspective you are looking at.

        NOTE: ensure the arguments are in the same direction! (Input/Output)

        Returns: 
        - A stacked plot of cell type of interest in a direction 
        """
    
    if direction == 'input': 
        type = 'type_pre'
        x ='bodyId_post_index'
        y='pcnt_ct_total_ct'
        parent = 'bodyId_post'
        y_title=f'Pre-Cell → ' +  parent_name + '(% of Total Connections)'

    elif direction == 'output':
        type = 'type_post'
        x ='bodyId_pre_index'
        y='pcnt_ct_total_ct'
        parent = 'bodyId_pre'
        y_title=f'Post-Cell → ' +  parent_name + '(% of Total Connections)'
    else:
        raise ValueError(f"Direction '{direction}' not recognized")
    
    drop_dup = celltype_df[celltype_df[type].isin(all_celltype[:num_celltype_in_plot])].copy()  
    drop_dup.loc[:,x] = drop_dup[parent].astype("category").cat.codes 

    stacked_plot = px.bar(
        drop_dup,
        x=x,
        y=y,
        color=type,
        barmode='stack',
        hover_data=[x, type],
        category_orders={
            type: all_celltype[:num_celltype_in_plot]},
        labels={
            x: parent_name + ' neuron (indexed)',
            y: y_title
        },
    )
    # Make background white and remove gridlines
    stacked_plot.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return stacked_plot

def get_all_parent_top_cts(random_celltypes_list, num_of_cells):
    """
    Get top values for major ct in OL.
        Arguments:
        - random_celltypes_list: dictionary of major ct in OL
        - num_of_cells: number of cell types to include in elbow calculation of elbow for a certain Parent cell type (20).
        Returns: 
        - concat_top_input_output: Input and Output values from all ct with their groups. 
    """
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    partner_folder = "partner_data_28_04_2026"

    main_input = []
    main_output = []

    for i in random_celltypes_list:
        # Input
        filename_in = f"input_{i}_df.pkl"
        filepath_in = data_dir / partner_folder / filename_in
        with open(filepath_in, "rb") as f:
            og_input = pickle.load(f)
            og_input_df = og_input.copy()
        _, _, weighted_input, input_to_celltype_df = pre_HP.get_input_connections_processing(og_input_df, num_of_cells) 
        
        
        input_to_celltype_df['Parent_cell_type'] = i
        input_to_celltype_df['top_ct'] = input_to_celltype_df['type_pre']
        group = random_celltypes_list[i]
        input_to_celltype_df.insert(loc=1, column='Group', value=group)

        # rank
        input_to_celltype_df = input_to_celltype_df[input_to_celltype_df['top_ct'].isin(weighted_input)] 
        weighted_rank = pd.Series(range(1, len(weighted_input)+1), index=weighted_input)
        input_to_celltype_df['weighted_median_rank'] = input_to_celltype_df['top_ct'].map(weighted_rank)

        # Global computation
        for u in input_to_celltype_df['bodyId_post'].unique():
            internal_rank = input_to_celltype_df[input_to_celltype_df['bodyId_post'] == u].groupby('top_ct')['weighted_pcnt_ct_total_ct'].median().sort_values(ascending = False).index.tolist() 
            map_rank = pd.Series(range(1, len(internal_rank)+1), index=internal_rank)
            input_to_celltype_df.loc[input_to_celltype_df['bodyId_post'] == u, 'internal_median_rank'] = (input_to_celltype_df.loc[input_to_celltype_df['bodyId_post'] == u, 'top_ct'].map(map_rank))
        
        #average this internal rank for each top cell type in the parent 
        input_to_celltype_df['avg_internal_rank_per_ct'] = input_to_celltype_df.groupby('top_ct')['internal_median_rank'].transform('mean').round(2)
        
        main_input.append(input_to_celltype_df)

        # Output
        filename_out = f"output_{i}_df.pkl"
        filepath_out = data_dir / partner_folder / filename_out
        with open(filepath_out, "rb") as f:
            og_output = pickle.load(f)
            og_output_df = og_output.copy()
        _, _, weighted_output, output_to_celltype_df = get_output_connections_processing(og_output_df, num_of_cells)
        
        
        output_to_celltype_df['Parent_cell_type'] = i
        output_to_celltype_df['top_ct'] = output_to_celltype_df['type_post']
        output_to_celltype_df.insert(loc=1, column='Group', value=group)

        # Weighted rank
        output_to_celltype_df = output_to_celltype_df[output_to_celltype_df['top_ct'].isin(weighted_output)] 
        weighted_rank = pd.Series(range(1, len(weighted_output)+1), index=weighted_output)
        output_to_celltype_df['weighted_median_rank'] = output_to_celltype_df['top_ct'].map(weighted_rank)

        # Global computation
        for u in output_to_celltype_df['bodyId_pre'].unique():
            internal_rank = output_to_celltype_df[output_to_celltype_df['bodyId_pre'] == u].groupby('top_ct')['weighted_pcnt_ct_total_ct'].median().sort_values(ascending = False).index.tolist() 
            map_rank = pd.Series(range(1, len(internal_rank)+1), index=internal_rank)
            output_to_celltype_df.loc[output_to_celltype_df['bodyId_pre'] == u, 'internal_median_rank'] = (output_to_celltype_df.loc[output_to_celltype_df['bodyId_pre'] == u, 'top_ct'].map(map_rank))
        
        #average this internal rank for each top cell type in the parent 
        output_to_celltype_df['avg_internal_rank_per_ct'] = output_to_celltype_df.groupby('top_ct')['internal_median_rank'].transform('mean').round(2)

        main_output.append(output_to_celltype_df)

    main_stats_input = pd.concat(main_input, ignore_index=True)
    main_stats_input['direction'] = 'input'

    main_stats_output = pd.concat(main_output, ignore_index=True)
    main_stats_output['direction'] = 'output'

    concat_top_input_output = pd.concat([main_stats_input, main_stats_output], axis=0, ignore_index=True)
    concat_top_input_output = concat_top_input_output.drop(columns=['bodyId_pre', 'bodyId_post', 'roi','sum_weight_by_postId', 'pcnt_indv_postsyn_total_postsyn', 'sum_typepre_to_postId','num_typepres_to_postId', 'pcnt_ct_total_ct', 'med_syn_typepre_to_post', 'avg_num_typepres_to_postId', 'num_neurons_top_ct_missing', 'med_num_typepres_to_postId', 'sum_weight_by_preId', 'pcnt_indv_presyn_total_presyn',
       'sum_typepost_from_preId', 'num_typeposts_from_preId',
       'med_syn_typepost_from_pre', 'avg_num_typeposts_to_preId',
       'med_num_typeposts_to_preId'])

    return concat_top_input_output

#  PLot all top cells 
def plot_top_parent_ct(group_map, concat_top_input_output):
    """
    Get top values for major ct in OL. Gives percent top cell type synw/ all synw for each parent cell type. This is done for every bodyId. 
        Arguments:
        -  group_map: Give a list  ['ONIN', 'ONCN', 'VPN', "VCN]
        - concat_top_input_output: dataframe of input and output values from randome celltype list
        Returns: 
        - elbow_stats_concat: Input and Output values from all ct with their groups. 
    """
    direction_map = {'input': 'green', 'output': 'blue'}
    top_plot = make_subplots(rows=1, cols=len(group_map), subplot_titles=group_map)

    for i, group in enumerate(group_map):
        ordered_ct = concat_top_input_output[(concat_top_input_output['Group'] == group) & (concat_top_input_output['direction'] == 'input')].groupby('Parent_cell_type')['bodyId_post'].nunique().sort_values(ascending=False).index.to_list()
        for direction in direction_map:
            subset_df = concat_top_input_output[(concat_top_input_output['Group'] == group) & (concat_top_input_output['direction'] == direction)]

            hover_text = (
                    'bodyId_pre: ' + subset_df['bodyId_pre'].astype(str) +  '<br>' +
                    'bodyId_post: ' + subset_df['bodyId_post'].astype(str) +  '<br>' +
                    'parent type: ' + subset_df['Parent_cell_type'].astype(str) +  '<br>' +
                    'Top Cell: ' + subset_df['top_ct'].astype(str) +  '<br>' +
                    'Pcnt of top cell type/ total cell types to parent: ' + subset_df['pcnt_ct_total_ct'].round(2).astype(str))

            top_plot.add_trace(
                go.Box(
                    x=subset_df['Parent_cell_type'],
                    y=subset_df['pcnt_ct_total_ct'],
                    boxpoints=None,
                    text=hover_text,
                    hoverinfo='text',
                    marker_color=direction_map[direction],
                    name=direction,
                    showlegend=True
                ),
                row=1, col=1+i
            )
            top_plot.update_yaxes(title_font={'size': 10}, range=[0, 100], title_text='Pcnt of top cell type/ total cell types to parent', row=1, col=1+i)
            top_plot.update_xaxes(categoryorder='array', categoryarray=ordered_ct, tickfont={'size': 10}, row=1, col=1+i)

            top_plot.update_layout(
                height=600,
                width=2000,
                boxmode='group',
                plot_bgcolor='white'
            )
    return top_plot
        
# Pivot_df (for hierarchical sorting)
def hierarchical_sort_df(df5):
    """" 
    This function is the pre-process df (all partner to parent in both directions)
    and df5 (top 5 partner to parent in both directions) for hierarchical sortig in the top_ct bubble and box plots
       
    Arguments:
       df and df5 dataframes
          
    Returns:
        pivot_df
   """
    # Introduce clip to set bounds for hierarchical sorting (happens inside plot_box_top_ct_coverage and plot_bubble_top_ct_coverage function)

    df5['avg_internal_rank_per_ct_capped'] = df5['avg_internal_rank_per_ct']
    df5.loc[df5['weighted_median_rank'] == 1, 'avg_internal_rank_per_ct_capped'] = (df5.loc[df5['weighted_median_rank'] == 1, 'avg_internal_rank_per_ct_capped'].clip(upper=1.25))
    df5.loc[df5['weighted_median_rank'] == 2, 'avg_internal_rank_per_ct_capped'] = (df5.loc[df5['weighted_median_rank'] == 2, 'avg_internal_rank_per_ct_capped'].clip(upper=2.5))

    pivot_df = (df5.drop_duplicates(subset=['Parent_cell_type', 'direction', 'top_ct', 'weighted_median_rank'])[['Parent_cell_type', 'Group', 'direction', 'top_ct', 'weighted_median_rank', 'avg_internal_rank_per_ct_capped']]) \
            .pivot_table(index=['Group', 'Parent_cell_type', 'direction'], columns='weighted_median_rank', values=['avg_internal_rank_per_ct_capped', 'top_ct'], aggfunc='first')\
            .reset_index()
    pivot_df.columns.names = (None,None)

    return pivot_df

#  Same plot as above but all optic lobe gorups in same plot
def plot_box_top_ct_coverage_allgroups(df5, sort_hier_df, direction): 
    """
    Gives median top X percent top cell type synw/ all synw for each parent cell type. This is done for every bodyId. The size of the dots indicate percent of bodyIds this top value is found.
        Arguments:
        - df5: dataframe of input and output values from randome celltype list
        - sort_hier_df: datafraem to use when sorting parent cell types by hierarchical sorting using internal rank capped
        - direction: 'input' or 'output'

        Returns: 
        - box plot for direction specified 
    """
     #  Remove weird data such as ["R1-R6", "L1", "L2", "L3", "L4", "L5", "T1"]
    if direction == 'input':
        to_drop = ["L1", "L2", "L3", "L4", "L5", "T1","R1-R6", "R7d" , "R8d", "R8p", "R7p"]
    elif direction == 'output':
        to_drop = ["R1-R6", "R7d" , "R8d", "R8p", "R7p"]
    else:
        to_drop = []

    df5 = df5[~df5["Parent_cell_type"].isin(to_drop)]
    sort_hier_df = sort_hier_df[~sort_hier_df["Parent_cell_type"].isin(to_drop)]

    df5 = df5[(df5['direction'] == direction)]
    sort_hier_df = sort_hier_df[sort_hier_df['direction'] == direction]

    # list of cts and numbr of neurons in ct
    ct_counts = df5[df5['direction'] == direction].groupby('Parent_cell_type')['num_neurons_parent_ct'].median().sort_values(ascending = False) 

    # Lists and dictionaries 
    rank={1: 'red', 2: 'orange', 3: 'tan', 4: 'skyblue', 5: 'blue'}
    
    # Plot
    top_box = go.Figure()
    rank_cols = [('avg_internal_rank_per_ct_capped', 1),('avg_internal_rank_per_ct_capped', 2),('avg_internal_rank_per_ct_capped', 3),('avg_internal_rank_per_ct_capped', 4),('avg_internal_rank_per_ct_capped', 5)]
    hierarchical_ir_ct = sort_hier_df.sort_values(by=rank_cols, ascending=[True, True, True, True, True])['Parent_cell_type'].unique().tolist() #by binnng first, second, third ranks into groups and capping   
    for r in rank:
        subset_df = df5[df5['weighted_median_rank'] == r]
        hover_text = (
                'Parent type: ' + subset_df['Parent_cell_type'].astype(str) +  '<br>' +
                'Top Cell: ' + subset_df['top_ct'].astype(str) +  '<br>' +
                'Median: ' + subset_df['median_pcnt_ct_total_ct'].round(2).astype(str) +  '<br>' +	
                'Weighted median based rank (all ct): ' + subset_df['weighted_median_rank'].astype(str) +  '<br>' +
                'Internal rank per cell of parent (avg across ct): '  + subset_df['avg_internal_rank_per_ct'].astype(str) +  '<br>' )

        top_box.add_trace(
            go.Box(
                x=subset_df['Parent_cell_type'],
                y=subset_df['weighted_pcnt_ct_total_ct'],  #change to weighted  median for consistency with ranking
                text=hover_text,
                hoverinfo='text',
                boxpoints='outliers',
                marker_color=rank[r],
                showwhiskers=False,
                sizemode='quartiles',
                name=r,
                # marker={'size': }
                showlegend=True
            )
        )
    top_box.add_trace(
        go.Scatter(
            x=ct_counts.index,
            y=ct_counts.values,
            mode='markers',
            marker=dict(size=10, color='black'),
            name='# Parent neurons',
            yaxis='y2'   # maps to secondary axis
        )
    )
    top_box.update_yaxes(title_font={'size': 10}, range=[0, 100], title_text=' weighted pcnt of cell type/ total cell types ' + direction + ' to parent')
    top_box.update_xaxes(categoryorder='array', categoryarray= hierarchical_ir_ct, tickfont={'size': 10}, title_text=' Parent type (ordered smallest --> largest by internal rank (hierarchical))')
    top_box.update_layout(boxmode='group', plot_bgcolor='white', title= "Parent-Partner in direction " + direction, yaxis2=dict(title='# parent neurons', overlaying='y', side='right', title_font={'size': 10}, range=[1,3], type='log'))

    return top_box, hierarchical_ir_ct

def top_ct_coverage_df(df5, sort_hier_df, NT_all_cts, direction):
    """
    Gives median top X percent top cell type synw/ all synw for each parent cell type. This is done for every bodyId. The size of the dots indicate percent of bodyIds this top value is found.
    The shape indicate group 
    Color is NT from def nt_celltype(random_celltypes_list):

        Arguments:
        - df5: dataframe of input and output values from randome celltype list
        - direction: 'input' or 'output'
        - sort_hier_df: dataframe to use when sorting parent cell types by hierarchical sorting using internal rank capped
        - NT_cells: df of all OL celltypes with their group and NT

        Returns: 
        - parent_nt_df, median_df_plot, hierarchical_ir_ct
    """
    # --------------------------- Preprocess data --------------------------------------
    #  Remove weird data such as ["R1-R6", "L1", "L2", "L3", "L4", "L5", "T1"]
    if direction == 'input':
        to_drop = ["L1", "L2", "L3", "L4", "L5", "T1","R1-R6", "R7d" , "R8d", "R8p", "R7p"]
    elif direction == 'output':
        to_drop = ["R1-R6", "R7d" , "R8d", "R8p", "R7p"]
    else:
        to_drop = []

    df5 = df5[~df5["Parent_cell_type"].isin(to_drop)]
    sort_hier_df = sort_hier_df[~sort_hier_df["Parent_cell_type"].isin(to_drop)]

    subset_df = df5[df5['direction'] == direction]
    sort_hier_df = sort_hier_df[sort_hier_df['direction'] == direction]
        
    # Computations 
    subset_df['weighted_median_rank'] = subset_df['weighted_median_rank'].astype(str)
    subset_df = subset_df[subset_df['weighted_median_rank'].isin(['1', '2', '3', '4', '5'])]
    subset_df.loc[:, 'pcnt_parent_neurons_top_ct_seen'] = subset_df['frac_parent_neurons_top_ct_seen'] * 100 

    #--------------------------- Lists and dictionaries ---------------------------
    rank_cols = [('avg_internal_rank_per_ct_capped', 1),('avg_internal_rank_per_ct_capped', 2),('avg_internal_rank_per_ct_capped', 3),('avg_internal_rank_per_ct_capped', 4),('avg_internal_rank_per_ct_capped', 5)]
    hierarchical_ir_ct = sort_hier_df.sort_values(by=rank_cols, ascending=[True, True, True, True, True])['Parent_cell_type'].unique().tolist() #by binnng first, second, third ranks into groups and capping 

    #--------------------------- Merge NT to Subset_df  ---------------------------
    # Merge on top_ct
    subset_df = subset_df.merge(NT_all_cts[['cell_type', 'dominant_nt', 'Group', 'nt_color', 'group_symbol', 'group_color']], left_on='top_ct', right_on='cell_type', how='left')
    subset_df = subset_df.rename(columns={'dominant_nt': 'top_ct_nt','Group_y': 'top_ct_group','nt_color': 'top_ct_nt_color','group_symbol': 'partner_group_symbol', 'Group_x': 'Group', 'group_color': 'partner_group_color'})
    subset_df = subset_df.drop(columns=['cell_type'])

    # Merge w Parent 
    subset_df = subset_df.merge(NT_all_cts[['cell_type', 'dominant_nt', 'nt_color', 'group_symbol', 'group_color']],left_on='Parent_cell_type',right_on='cell_type',how='left')
    subset_df = subset_df.rename(columns={'dominant_nt': 'nt_parent','nt_color': 'parent_nt_color','group_symbol': 'parent_group_symbol', 'group_color': 'parent_group_color'})
    subset_df = subset_df.drop(columns=['cell_type'])

    # Celltypes outside optic lobe NT color and symbol
    subset_df['top_ct_group'] = subset_df['top_ct_group'].fillna('outside optic lobe')
    subset_df['top_ct_nt'] = subset_df['top_ct_nt'].fillna('unknown')
    subset_df['top_ct_nt_color'] = subset_df['top_ct_nt_color'].fillna('grey')
    subset_df['partner_group_symbol'] = subset_df['partner_group_symbol'].fillna('circle-open')

    # df for plot (weighted median per top_ct) to ease with comuptaiton power
    median_df_plot = subset_df.drop_duplicates(subset=['Parent_cell_type', 'top_ct','weight_median_pcnt_ct_total_ct'], keep='first')

    # Paret df for y=0 parent Nt adn Group 
    parent_nt_df = (
        subset_df[['Parent_cell_type', 'parent_nt_color', 'parent_group_symbol' , 'parent_group_color', 'nt_parent',  'Group']]
        .drop_duplicates()
    )
    parent_nt_df['hover_text']= (
                'Parent type: ' + parent_nt_df['Parent_cell_type'] + '<br>' +
                'Parent NT: ' + median_df_plot['nt_parent'] + '<br>' +
                'Parent Group: ' + median_df_plot['Group']
            )
    
    return parent_nt_df, median_df_plot, hierarchical_ir_ct
def plot_bubble_top_ct_coverage_allgroups_combined(parent_nt_df, median_df_plot, hierarchical_ir_ct, direction): 
    """
    Gives median top X percent top cell type synw/ all synw for each parent cell type. This is done for every bodyId. The size of the dots indicate percent of bodyIds this top value is found.
    The shape indicate group 
    Color is NT from def nt_celltype(random_celltypes_list):

        Arguments:
        parent_nt_df, median_df_plot, hierarchical_ir_ct

        Returns: 
        - bubble plot (with number of cells top ct is NOT in) for direction specified
    """

    # Plot
    bubble_plot = go.Figure()
    bubble_plot.add_trace(
        go.Scatter(
            x=parent_nt_df['Parent_cell_type'],
            y=[0] * len(parent_nt_df),
            mode='markers',
            marker=dict(
                size=12,
                color=parent_nt_df['parent_nt_color'],
                symbol=parent_nt_df['parent_group_symbol'], 
                line=dict(width=1, color='black')
            ),
            name='Parent_cell_type',
            hoverinfo='text',
            text=parent_nt_df['hover_text'],
            showlegend=False
        ))

    median_df_plot['hover_text'] = (
        'Parent type: ' + median_df_plot['Parent_cell_type'].astype(str) + '<br>' +
        'Top Cell: ' + median_df_plot['top_ct'].astype(str) + '<br>' +
        'Weighted median: ' + median_df_plot['weight_median_pcnt_ct_total_ct'].round(2).astype(str) + '<br>' +
        'Weighted median rank: ' + median_df_plot['weighted_median_rank'].astype(str) + '<br>' +
        'Internal rank (avg): ' + median_df_plot['avg_internal_rank_per_ct'].astype(str) + '<br>' +
        'Parent NT: ' + median_df_plot['top_ct_nt'] + '<br>' +
        'Parent Group: ' + median_df_plot['top_ct_group'] + '<br>' +
        '% Parent neurons seen: ' + median_df_plot['pcnt_parent_neurons_top_ct_seen'].round(1).astype(str)
    )

    for r in ['1', '2', '3', '4', '5']:
        plot_df = median_df_plot[median_df_plot['weighted_median_rank'] == r]
        bubble_plot.add_trace(
            go.Scatter(
                x=plot_df['Parent_cell_type'],
                y=plot_df['weight_median_pcnt_ct_total_ct'],
                mode='markers',
                text=plot_df['hover_text'],
                hoverinfo='text',
                marker=dict(
                    size=12,
                    color=plot_df['top_ct_nt_color'],
                    symbol=plot_df['partner_group_symbol'],
                    line=dict(width=1, color='black')
                ),
                name=f'Rank {r}',     
                showlegend=True       
            )
        )
    bubble_plot.update_yaxes(title_font={'size': 10}, range=[-5, 100], title_text=' weighted pcnt / total cell type ' + direction + ' to parent')
    bubble_plot.update_xaxes(categoryorder='array', categoryarray= hierarchical_ir_ct, tickfont={'size': 10}, title_text=' Parent type (ordered smallest --> largest by internal rank (hierarchical))')
    bubble_plot.update_layout(boxmode='group', plot_bgcolor='white', title= "Parent-Partner in direction " + direction)
    bubble_plot.add_hline(y=0,line_width=1,line_dash="dot",line_color="gray")

    return bubble_plot
def plot_bubble_top_ct_coverage_allgroups_subplots(parent_nt_df, median_df_plot, hierarchical_ir_ct, direction): 
    """
    Gives median top X percent top cell type synw/ all synw for each parent cell type. This is done for every bodyId. The size of the dots indicate percent of bodyIds this top value is found.
    The shape indicate group 
    Color is NT from def nt_celltype(random_celltypes_list):

        Arguments:
        - parent_nt_df, median_df_plot, hierarchical_ir_ct

        Returns: 
        - bubble plot for direction specified
    """
    # Plot
    bubble_plot = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Neurotransmitter", "Group"])

    # ---- Parent NT (row 1) ----
    bubble_plot.add_trace(
        go.Scatter(
            x=parent_nt_df['Parent_cell_type'],
            y=[0]*len(parent_nt_df),
            mode='markers',
            marker=dict(
                size=12,
                color=parent_nt_df['parent_nt_color'],
                line=dict(width=1, color='black')
            ),
            hovertext=parent_nt_df['hover_text'],
            hoverinfo='text',
            showlegend=True
        ),
        row=1, col=1
    )

    # ---- Parent Group (row 2) ----
    bubble_plot.add_trace(
        go.Scatter(
            x=parent_nt_df['Parent_cell_type'],
            y=[0]*len(parent_nt_df),
            mode='markers',
            marker=dict(
                size=12,
                color=parent_nt_df['parent_group_color'],
                line=dict(width=1, color='black')
            ),
            hovertext=parent_nt_df['hover_text'],
            hoverinfo='text',
            showlegend=True
        ),
        row=2, col=1
    )

    median_df_plot['hover_text'] = (
        'Parent type: ' + median_df_plot['Parent_cell_type'].astype(str) + '<br>' +
        'Top Cell: ' + median_df_plot['top_ct'].astype(str) + '<br>' +
        'Weighted median: ' + median_df_plot['weight_median_pcnt_ct_total_ct'].round(2).astype(str) + '<br>' +
        'Weighted median rank: ' + median_df_plot['weighted_median_rank'].astype(str) + '<br>' +
        'Internal rank (avg): ' + median_df_plot['avg_internal_rank_per_ct'].astype(str) + '<br>' +
        'Parent NT: ' + median_df_plot['top_ct_nt'] + '<br>' +
        'Parent Group: ' + median_df_plot['top_ct_group'] + '<br>' +
        '% Parent neurons seen: ' + median_df_plot['pcnt_parent_neurons_top_ct_seen'].round(1).astype(str)
    )

    for r in ['1','2','3','4','5']:
        plot_df = median_df_plot[median_df_plot['weighted_median_rank'] == r]

        # NT subplot
        bubble_plot.add_trace(
            go.Scatter(
                x=plot_df['Parent_cell_type'],
                y=plot_df['weight_median_pcnt_ct_total_ct'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=plot_df['top_ct_nt_color'],
                    line=dict(width=1, color='black')
                ),
                hovertext=plot_df['hover_text'],
                hoverinfo='text',
                name=f'Rank {r}',
                showlegend=True
            ),
            row=1, col=1
        )

        # Group subplot
        bubble_plot.add_trace(
            go.Scatter(
                x=plot_df['Parent_cell_type'],
                y=plot_df['weight_median_pcnt_ct_total_ct'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=plot_df['partner_group_color'],
                    line=dict(width=1, color='black')
                ),
                hovertext=plot_df['hover_text'],
                hoverinfo='text',
                name=f'Rank {r}',
                showlegend=True
            ),
            row=2, col=1
        )

    bubble_plot.update_yaxes(title_font={'size': 10}, range=[-5, 100], title_text=' weighted pcnt / total cell type ' + direction + ' to parent',  row=1, col=1)
    bubble_plot.update_yaxes(title_font={'size': 10}, range=[-5, 100], title_text=' weighted pcnt / total cell type ' + direction + ' to parent',  row=2, col=1)
    bubble_plot.update_xaxes(categoryorder='array', categoryarray= hierarchical_ir_ct, tickfont={'size': 10}, title_text=' Parent type (ordered smallest --> largest by internal rank (hierarchical))', row=2, col=1)
    bubble_plot.update_xaxes(showticklabels=True, row=1, col=1)
    bubble_plot.update_layout(boxmode='group', plot_bgcolor='white', title= "Parent-Partner in direction " + direction)
    bubble_plot.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    bubble_plot.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    return bubble_plot


# log10(# of parents/ # of partners from a cell types)
def parent_by_partner_log_df(all_ol_ct, bipartite_parent_input_v3, bipartite_parent_output_v3):
    """
    Note this will only be cut by reliability from parent perspective nto partner. 
    This only gives info on partner number of cells (nothing more) for all OL_R cts. 
    Refer to input_bipartite_table_v3.pkl/ output_bipartite_table_v3.pkl files for pcnt info of partner (only covers cts >= 10)

    all_input_monopartite_table/ output version = is for all OL celltypes (700)

    Note: this is using all OL celltypes as parent list
    """
    # Prep path 
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])

    input = "all_OL_input_monopartite_table_v3.pkl"
    filepath_input = data_dir/ input
    if filepath_input.exists():
            with open(filepath_input, "rb") as f:
                df_input_all_ol_parents = pickle.load(f)

    output = "all_OL_output_monopartite_table_v3.pkl"
    filepath_output = data_dir / output
    if filepath_output.exists():
            with open(filepath_output, "rb") as f:
                df_output_all_ol_parents = pickle.load(f)

    all_input_monopartite_table_v3 = df_input_all_ol_parents.copy(deep=True)
    all_output_monopartite_table_v3 = df_output_all_ol_parents.copy(deep=True)
    OL_greater_than_10_bipartite_in = bipartite_parent_input_v3.copy(deep=True)
    OL_greater_than_10_bipartite_out = bipartite_parent_output_v3.copy(deep=True)

    # ONLY COnsider partners for CUTOFF (this cutoff was at 3 percent and 0.8 reliability) (no partner reliability here!)
    all_input_monopartite_table_v3_cutoff = all_input_monopartite_table_v3[all_input_monopartite_table_v3['cutoff'] == 1].copy()
    all_output_monopartite_table_v3_cutoff = all_output_monopartite_table_v3[all_output_monopartite_table_v3['cutoff'] == 1].copy()

    # Assign number of neurons to partner (input parent)
    all_ol_ct = all_ol_ct[all_ol_ct['Group'] != 'other']
    cell_map = dict(zip(all_ol_ct['cell_type'], all_ol_ct['no. of cells']))
    all_input_monopartite_table_v3_cutoff['total_partner_neurons_2'] = all_input_monopartite_table_v3_cutoff['partner_cell'].map(cell_map)
    all_output_monopartite_table_v3_cutoff['total_partner_neurons_2'] = all_output_monopartite_table_v3_cutoff['partner_cell'].map(cell_map)

    # find log parent/ partner num neuron raito
    all_input_monopartite_table_v3_cutoff["log_ratio"] = np.log10(all_input_monopartite_table_v3_cutoff["num_neurons_parent_ct"] / all_input_monopartite_table_v3_cutoff["total_partner_neurons_2"])
    all_output_monopartite_table_v3_cutoff["log_ratio"] = np.log10(all_output_monopartite_table_v3_cutoff["num_neurons_parent_ct"] / all_output_monopartite_table_v3_cutoff["total_partner_neurons_2"])

    # remove all nan partners
    OL_output_monopartite_table_v3_cutoff = all_output_monopartite_table_v3_cutoff[all_output_monopartite_table_v3_cutoff['total_partner_neurons_2'].notna()]
    OL_input_monopartite_table_v3_cutoff = all_input_monopartite_table_v3_cutoff[all_input_monopartite_table_v3_cutoff['total_partner_neurons_2'].notna()]


    # For OL neurons with more than 10 cells thresholded (already cut-off at reliabilty and percent threshold)
    OL_greater_than_10_bipartite_in["log_ratio"] = np.log10(OL_greater_than_10_bipartite_in["num_neurons_parent_ct"] / OL_greater_than_10_bipartite_in["total_partner_neurons_2"])
    OL_greater_than_10_bipartite_out["log_ratio"] = np.log10(OL_greater_than_10_bipartite_out["num_neurons_parent_ct"] / OL_greater_than_10_bipartite_out["total_partner_neurons_2"])

    return OL_output_monopartite_table_v3_cutoff, OL_input_monopartite_table_v3_cutoff, OL_greater_than_10_bipartite_in, OL_greater_than_10_bipartite_out
def parent_by_partner_log_plot(df , partners_included, direction):
    """
    Arguments:

    df: all_input_monopartite_table_v3_cutoff, all_output_monopartite_table_v3_cutoff, notna_all_output_monopartite_table_v3_cutoff, notna_all_input_monopartite_table_v3_cutoff 
    partners_included: df and this argument should match on user end. 
        - if partners_included = 'all' then df should be all_*
        = if partners_included = 'ol' then df should be notna*

    Return:
    fig
    """
    # make in plotly go and color by number of neuron invlved in making the plot eg : 10 parent / 10 partners etc
    if partners_included == 'ol':
        add = " (only optic lobe partners included)"
        bins = [0, 10, 50, 100, 250, 500, 800, 900, float("inf")]
        labels = ["1-10", "11-50", "51-100", "101-250", "251-500", "501-800", "801-900", "1000+"]
    if partners_included == 'ol_10':
        add = " (only optic lobe celltype > 10 cells included (all thesholded partners considered))"
        bins = [10, 50, 100, 250, 500, 800, 900, float("inf")]
        labels = ["10-50", "51-100", "101-250", "251-500", "501-800", "801-900", "1000+"]
    if partners_included == 'all':
        add = " All partners"
        bins = [0, 10, 50, 100, 250, 500, 800, 900, float("inf")]
        labels = ["1-10", "11-50", "51-100", "101-250", "251-500", "501-800", "801-900", "1000+"]

    df["parent_size_bin"] = pd.cut(df["num_neurons_parent_ct"], bins=bins, labels=labels)

    # Plot histogram
    fig = px.histogram(
        df,
        x="log_ratio",
        color="parent_size_bin",  
        nbins=40,
        opacity=0.6
    )

    fig.update_layout(
        title="Number of Parent and Partner neurons ratio in direction: " + direction + add,
        xaxis_title="log_ratio (number of parent neurons / number of partner neurons)",
        yaxis_title="Count",
        barmode='stack',
        plot_bgcolor='white', paper_bgcolor='white')

    fig.show()
def parent_by_partner_log_boxplot(df, direction):
    """
    Arguments:
    df: all_input_monopartite_table_v3_cutoff, all_output_monopartite_table_v3_cutoff, notna_all_output_monopartite_table_v3_cutoff, notna_all_input_monopartite_table_v3_cutoff 

    Return:
    fig
    """
    sort_parent = df.groupby('parent')["num_neurons_parent_ct"].max().sort_values(ascending=False)

    # Plot histogram
    fig = px.scatter(
        df,
        x="parent",
        y="log_ratio",
        category_orders={"parent": sort_parent.index.tolist()},
        hover_data=['partner_cell']
    )

    fig.update_layout(
        title="Number of Parent and Partner neurons ratio in direction: " + direction,
        xaxis_title="Parent Ordered by Size",
        yaxis_title="log_ratio (number of parent neurons / number of partner neurons)",
        plot_bgcolor='white', paper_bgcolor='white')

    fig.show()