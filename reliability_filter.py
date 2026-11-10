import pandas as pd
from neuprint import Client
from neuprint import fetch_adjacencies, NeuronCriteria as NC, fetch_synapses, SynapseCriteria as SC, fetch_neurons
import numpy as np
import plotly.express as px
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pre_HP
import inspect
import yaml
from pathlib import Path
import pickle

# PARENT RELIABILTIY
# Explore a cut-off using relibability and weighted median criteria 
def median_frac_plot_func(direction, parent_name, plot_type, celltype_df, x_cutoff, y_cutoff):
    """
    Plot the relationship between the median percent of partner cell types
    and the fraction of parent cells connected, for input or output connections.Get df_plot form cut-off function
    
    Parameters:
    - direction: 'input' or 'output'
    - parent_name: str, the parent cell type name (e.g., 'Dm19')
    - plot_type: polar or cartesian
    
    Returns:
    - Plotly Figure
    """
    if direction == 'input':
        celltype_df = celltype_df.drop_duplicates(subset=['type_pre'], keep='first')
        hover_data = ['type_pre', 'weight_median_pcnt_ct_total_ct', 'frac_parent_neurons_top_ct_seen']
        color_discrete_sequence = ['green']

    elif direction == 'output':
        celltype_df = celltype_df.drop_duplicates(subset=['type_post'], keep='first')
        hover_data = ['type_post', 'weight_median_pcnt_ct_total_ct', 'frac_parent_neurons_top_ct_seen']
        color_discrete_sequence = ['blue']

    else:
        raise ValueError("direction must be 'input' or 'output'")
    
    x = celltype_df['weight_median_pcnt_ct_total_ct']
    y = celltype_df['frac_parent_neurons_top_ct_seen']

    if plot_type == 'polar':
        celltype_df['r'] = np.sqrt(x**2 + y**2)
        celltype_df['theta'] = np.arctan2(y, x)

        fig = px.scatter(
            celltype_df,
            x='r',
            y='theta',
            hover_data=hover_data,
            color_discrete_sequence=color_discrete_sequence,
            title=f"{parent_name} ({direction}) Polar Plot"
        )


    elif plot_type == 'cartesian':
        fig = px.scatter(
            celltype_df,
            x='weight_median_pcnt_ct_total_ct',
            y='frac_parent_neurons_top_ct_seen',
            hover_data=hover_data,
            color_discrete_sequence=color_discrete_sequence,
            # title=f"{parent_name} ({direction}) Cartesian Plot"
        )
        # Add vertical cutoff line
        fig.add_vline(x=x_cutoff, line_dash="dash", line_color="red")

        # Add horizontal cutoff line
        fig.add_hline(y=y_cutoff, line_dash="dash", line_color="red")

    else:
        raise ValueError("plot_type must be 'polar' or 'cartesian'")
    
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1400,
        height=900,
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(
            range=[0, None],
        ),
        yaxis=dict(
            range=[0, None],
        )
    )

    fig.update_xaxes(
    tickfont=dict(size=16, family="Arial")
)

    fig.update_yaxes(
        tickfont=dict(size=16, family="Arial")
    )
    return fig
def three_parameter_plot_func(direction, parent_name, celltype_df):
    """
    Plot median, distribution, and fraction of connected cell types for input/output connections.

    Parameters:
        direction: 'input' or 'output'
        parent_name: str, the parent cell type name (e.g., 'Dm19')

    Returns:
        elbow_plot: Plotly Figure
        elbow_df: DataFrame used for plotting
    """

    if direction not in ['input', 'output']:
        raise ValueError(f"Direction '{direction}' not recognized")

    if direction == 'input':
        x = 'type_pre'
        # used celltype_df for  y2 = 'pcnt_ct_total_ct'
        elbow_df = celltype_df.drop_duplicates(subset=[x], keep='first') # this is LVL 3 df, for  y1 = 'weight_median_pcnt_ct_total_ct'
        

    elif direction == 'output':
        x = 'type_post'
        # used celltype_df for  y2 = 'pcnt_ct_total_ct'
        elbow_df = celltype_df.drop_duplicates(subset=[x], keep='first') # this is LVL 3 df, for  y1 = 'weight_median_pcnt_ct_total_ct'

    # y variables
    y_median = 'weight_median_pcnt_ct_total_ct'     # median % ct/total (per-partner)
    y_pcnt = 'pcnt_ct_total_ct'                     # per-neuron % ct/total (distribution)
    y_frac = 'frac_parent_neurons_top_ct_seen'       # fraction connected (secondary y)
    plot_title = f"{parent_name}: % ct / total (median and distribution) and fraction ({direction})"


    # Sort elbow_df by median descending (this defines the desired x-order)
    elbow_df = elbow_df.sort_values(by=[y_median], ascending=False).reset_index(drop=True)

    # Force the per-neuron rows to use the same categorical ordering so box/violin respect it
    ordered_categories = elbow_df[x].tolist()
    celltype_df[x] = pd.Categorical(celltype_df[x], categories=ordered_categories, ordered=True)

    # PLot the elbow
    elbow_plot = make_subplots(specs=[[{"secondary_y": True}]])

    elbow_plot.add_trace(go.Scatter(
        x=elbow_df[x], # cell type partner
        y=elbow_df[y_median], # weighted pcnt ct total 
        mode='lines+markers',
        line={"width": 1},
        name="weight_median % ct/total",
        marker_color='darkgreen',
        showlegend=False
    ),
    secondary_y=False,)

    elbow_plot.add_trace(go.Box(
        x=celltype_df[x], # cell type partner
        y=celltype_df[y_pcnt], # pcnt ct total
        boxpoints='outliers',
        line={"width": 1},
        name = "% ct/total (distribution)",
        showlegend=False,
        marker_color='lightgreen'
        ),
        secondary_y=False,)

    elbow_plot.add_trace(
        go.Scatter(
            x=elbow_df[x],
            y=elbow_df[y_frac],
            mode='lines+markers',
            line=dict(width=1, color='purple'),
            name='Fraction connected',
            showlegend=False
        ),
        secondary_y=True,
    )

    elbow_plot.update_layout(
        # title={'text': plot_title},
        xaxis=dict(categoryorder='array',categoryarray=ordered_categories),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1400,
        height=900,
        xaxis_title=None,
        yaxis_title=None
    ) 
    elbow_plot.update_xaxes(range=[-1, 50], tickfont=dict(size=16, family="Arial"))   
    elbow_plot.update_yaxes(tickfont=dict(size=16, family="Arial"))                            
   
    return elbow_plot
# ALL FUNCTIONS COMBINED
def frac_with_median_ordering_summary(direction, parent_name, plot_type, x_cutoff=3, y_cutoff=0.8):
    """ Combined def pcnt_median_frac(direction, parent_name): and def med_vs_fraction(direction, parent_name):
    """
    _, celltype_df = frac_cell_pcnt_cutoff_v3(direction, parent_name, x_cutoff, y_cutoff)
    median_frac_plot= median_frac_plot_func(direction, parent_name, plot_type, celltype_df , x_cutoff, y_cutoff)
    three_parameter_plot= three_parameter_plot_func(direction, parent_name, celltype_df)


    return three_parameter_plot, median_frac_plot, celltype_df


# Cut-off Function (This is replacing the elbow!): This is population-level. based on frac > 0.8 and pcnt > 3 
def frac_cell_pcnt_cutoff_v3(direction, parent_name, x_cutoff=3, y_cutoff=0.8):
    """ This is to find a new cutoff. 
    Arguments:
    - parent_name: naem of parent e.g., LC10a 
    - direction: input or output 
    - x_cutoff: pcnt cut off (e.g: 1, 2, 0.1) default is 3
    - y_cutoff: reliability (e.g: 0.8, 0.1) default is 0.8
    Returns: 
    df_plot
    """
    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])
    partner_folder = "partner_data_28_04_2026"

    if direction == 'input':
        filename = f"input_{parent_name}_df.pkl"
        type_col = 'type_pre'
        filepath = data_dir / partner_folder / filename
        with open(filepath, "rb") as f:
            celltype_df = pickle.load(f)
        _, _, _, df = pre_HP.get_input_connections_processing(celltype_df) 
    elif direction == 'output':

        filename = f"output_{parent_name}_df.pkl"
        type_col = 'type_post'
        filepath = data_dir / partner_folder / filename
        with open(filepath, "rb") as f:
            celltype_df = pickle.load(f)
        _, _, _, df = pre_HP.get_output_connections_processing(celltype_df)
        

    else:
        raise ValueError("direction must be 'input' or 'output'")
    
    
    df_plot_og = df.drop_duplicates(subset=[type_col], keep='first')

    # Escape criteria partners
    partners_escape_criteria = ['R7y', 'R7p', 'R8y', 'R8p']
    
    df_below_threshold = df_plot_og[
        (~df_plot_og[type_col].isin(partners_escape_criteria)) &   # ← protect R7/R8
        (df_plot_og['weight_median_pcnt_ct_total_ct'] < x_cutoff) &
        (df_plot_og['frac_parent_neurons_top_ct_seen'] < y_cutoff)
    ].copy()

    partner_cells_remove = df_below_threshold[type_col].unique().tolist()
    initial_cut = df[~df[type_col].isin(partner_cells_remove)].copy()

    # remove 'R7y', 'R7p', 'R8y', 'R8p' below 1 pcnt and 0.4 reliability 
    remove_R7_R8_below_cutoff = (
        initial_cut[type_col].isin(partners_escape_criteria) &
        (initial_cut['weight_median_pcnt_ct_total_ct'] < 1) &
        (initial_cut['frac_parent_neurons_top_ct_seen'] < 0.4)
    )
    df_plot_cutoff = initial_cut[~remove_R7_R8_below_cutoff] # returns neuron ID to neuron ID level df

    return df_plot_cutoff, df


#  TEST FUNCTIONS (NOT MAIN): This is to find a good cutoff for the weighted pcercent to consider
#  This is to see which parents show partners with pcnt 3 < x < 10. This is to help decide cutoff (saved dict in connectome folder)
def find_best_cutoff(parent_list, x1=3, x2=10, y=0.8):
    """ This is to see if there are partners which 3< x < 10. These are problematic and need to be cut off.
    Arguments:

    Returns: 
    dict
    """
    cutoff_data = {}

    for parent_name in parent_list: 
   
        _, _, _, in_celltype_df = pre_HP.get_input_connections(parent_name) # celltype_df is the original dataframe 
        in_celltype_df = in_celltype_df.drop_duplicates(subset=['type_pre'], keep='first')

        df_not_threshold_in = in_celltype_df[
            (in_celltype_df['weight_median_pcnt_ct_total_ct'] > x1) &
            (in_celltype_df['weight_median_pcnt_ct_total_ct'] < x2) &
            (in_celltype_df['frac_parent_neurons_top_ct_seen'] < y)
        ]


        _, _, _, out_celltype_df = pre_HP.get_output_connections(parent_name)
        out_celltype_df = out_celltype_df.drop_duplicates(subset=['type_post'], keep='first')

        df_not_threshold_out = out_celltype_df[
            (out_celltype_df['weight_median_pcnt_ct_total_ct'] > x1) &
            (out_celltype_df['weight_median_pcnt_ct_total_ct'] < x2) &
            (out_celltype_df['frac_parent_neurons_top_ct_seen'] < y)
        ]
        
        # final dict
        cutoff_data[parent_name] = {
            "df_not_threshold_out": df_not_threshold_out,
            "df_not_threshold_in": df_not_threshold_in,
            "n_in": len(df_not_threshold_in),
            "n_out": len(df_not_threshold_out)
        }
                
    return cutoff_data
#  Number of parents which fall within 3 < x < 10 pcnt criteria
def plot_cutoff_counts_plotly(cutoff_data):

    # If your input is a dict, get keys
    parents = list(cutoff_data.keys())

    n_in = [cutoff_data[p]['n_in'] for p in parents]
    n_out = [-cutoff_data[p]['n_out'] for p in parents]  # negative axis

    fig_1 = go.Figure()

    fig_1.add_trace(go.Bar(
        x=parents,
        y=n_in,
        name="Input",
        hovertemplate="Parent: %{x}<br>n_in: %{y}<extra></extra>"
    ))

    fig_1.add_trace(go.Bar(
        x=parents,
        y=n_out,
        name="Output",
        hovertemplate="Parent: %{x}<br>n_out: %{y}<extra></extra>"
    ))

    fig_1.update_layout(
        title="Moderate-Strength (3-10%) & Low-Reliability (<0.8) Partners",
        xaxis_title="Parent Cell Type",
        yaxis_title="Number of Partners Cut",
        barmode='relative',
        template="plotly_white"
    )

    fig_1.update_yaxes(zeroline=True, zerolinewidth=2)
    fig_1.show()


    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Input", "Output"),
        shared_yaxes=True
    )

    for parent in parents:

        df_in = cutoff_data[parent]["df_not_threshold_in"]
        df_out = cutoff_data[parent]["df_not_threshold_out"]

        if not df_in.empty:
            fig2.add_trace(
                go.Scatter(
                    x=df_in["frac_parent_neurons_top_ct_seen"],
                    y=df_in["weight_median_pcnt_ct_total_ct"],
                    mode="markers",
                    marker=dict(color="green", size=8),
                    hovertemplate=(
                        "Parent: %{customdata[1]}<br>"
                        "type_pre: %{customdata[0]}<br>"
                        "Weighted Median %: %{y:.2f}<extra></extra>"
                    ),
                    customdata=df_in[["type_pre" , "type_post"]].values,
                    showlegend=False
                ),
                row=1, col=1
            )

        if not df_out.empty:
            fig2.add_trace(
                go.Scatter(
                    x=df_out["frac_parent_neurons_top_ct_seen"],
                    y=df_out["weight_median_pcnt_ct_total_ct"],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    hovertemplate=(
                        "Parent: %{customdata[1]}<br>"
                        "type_post: %{customdata[0]}<br>"
                        "Weighted Median %: %{y:.2f}<extra></extra>"
                    ),
                    customdata=df_out[["type_post" , "type_pre"]].values,
                    showlegend=False
                ),
                row=2, col=1
            )

    fig2.update_layout(
        title="Weighted Median Percent of Problematic Partners with the reliability (hover to see parent and partner)",
        template="plotly_white"
    )

    fig2.update_yaxes(title_text="Weighted Median Percent", row=1, col=1)
    fig2.update_yaxes(title_text="Weighted Median Percent", row=2, col=1)
    fig2.update_xaxes(title_text="Reliability Fraction", row=1, col=1)
    fig2.update_xaxes(title_text="Reliability Fraction", row=2, col=1)

    fig2.show()




# PARTNER RELIABILITY
def visualise_partner_reliability(bipart, direction):

    """
    This using the unfiltered bipartite dataframe which retains all partners despite reliability. 
    However parents are filtered by reliability (3, 0.8): (input_bipart / output_bipart)
    """

    # Reliability distribution among partners in output and input direction
    partner_freq = bipart.groupby('partner_cell').size().reset_index(name='freq')# this is just for ordering x axis 
    rank_order = partner_freq.sort_values('freq', ascending=False)['partner_cell'].tolist()
    reliability_fig = px.box(bipart, x="partner_cell", y="parent_reliability_2", category_orders={"partner_cell": rank_order})

    reliability_fig.update_layout(
        title={
            'text': f"Partner Reliability when Parents are Reliable (3 pcnt ; 0.8): {direction}"})

    return reliability_fig

def two_way_reliability_with_celltype_size_df(input_bipart, output_bipart , syn_partner_cutoff=None, syn_parent_cutoff=None, partner_reliability_cutoff=None):
    """
    This using the unfiltered bipartite dataframe which retains all partners despite reliability. 
    However parents are filtered by reliability (3, 0.8): (input_bipart / output_bipart)
    """
    df_in = input_bipart.copy()
    df_out = output_bipart.copy()

    def process_df(df):
        df["log_ratio"] = np.log10(df["num_neurons_parent_ct"] / df["total_partner_neurons_2"])
        df["log_raw_syn_nern_partner"] = np.log10(df["raw_syn_nern_partner_2"])
        df["log_raw_syn_nern"] = np.log10(df["raw_syn_nern"])
        df["log_raw_syn"] = np.log10(df["raw_syn"])
        df["log_avg_num_parents_contact_partner_2"] = np.log10(df["avg_num_parents_contact_partner_2"])
        df["log_avg_num_partners_contact_parent"] =  np.log10(df["avg_num_partners_contact_parent"])

        df["log_ratio_contact_pattern"]= np.log10(df["avg_num_partners_contact_parent"] / df["avg_num_parents_contact_partner_2"])
        if syn_partner_cutoff is not None: 
            df = df[(df["raw_syn_nern_partner_2"] > syn_partner_cutoff)]
        elif syn_parent_cutoff is not None:
            df = df[df["raw_syn_nern"] > syn_parent_cutoff]
        elif partner_reliability_cutoff is not None:
            df = df[df["parent_reliability_2"] >= partner_reliability_cutoff]
        elif syn_partner_cutoff and syn_parent_cutoff is not None:
            df = df[(df["raw_syn_nern_partner_2"] > syn_partner_cutoff) & (df["raw_syn_nern"] > syn_parent_cutoff)]
        
        return df

    df_in_bipart = process_df(df_in)
    df_out_bipart = process_df(df_out)

    return df_in_bipart, df_out_bipart

def two_way_reliability_with_celltype_size_plot(bipart, plot_type, direction, x_axis=None,  y_axis=None, color_axis=None): 
    """
    This using the unfiltered bipartite dataframe which retains all partners despite reliability. 
    However parents are filtered by reliability (3, 0.8): (input_bipart / output_bipart)
    args: reliability, contact_patttern, log_log, 3d_scatter
    """
    df = bipart.copy()
    color_scale = [
    [0.0, "lightgrey"],
    [0.5, "lightgrey"],
    [0.67, "yellow"],
    [0.83, "orange"],
    [1.0, "red"]
]
    if plot_type == 'reliability':
        fig = px.scatter(df, 
                            x=x_axis, 
                            y=y_axis, #"parent_reliability_2" or "partner_reliability"
                            hover_data=["parent", "partner_cell","num_neurons_parent_ct", "total_partner_neurons_2" ], 
                            color=color_axis, #"parent_reliability_2" or "partner_reliability"
                            color_continuous_scale=color_scale,
                            range_color=[0.4, 1],
                            render_mode="svg", 
                            labels={ "log_ratio": "log10 (# Parent CT / # Partner CT)",'parent_reliability_2': "Partner -> Parent Reliability (2)",'partner_reliability': "Parent -> Partner Reliability (1)" })
        fig.update_layout(
            showlegend = False,
            font=dict(
                family="Arial",
                size=16),
            # title={
            #     'text': f"#Parent / #Partner Log Ratio Plotted with Partner's Reliability of Parent: {direction}]"},        
            plot_bgcolor='white', paper_bgcolor='white',
            width=1400,
            height=900)
    
    if plot_type == 'contact_pattern':
        fig = px.scatter(df, 
                            x=x_axis, 
                            y=y_axis, #"log_avg_num_partners_contact_parent", "log_avg_num_parents_contact_partner_2"
                            hover_data=["parent", "partner_cell","num_neurons_parent_ct", "total_partner_neurons_2", "avg_num_partners_contact_parent", "avg_num_parents_contact_partner_2"], 
                            color=color_axis, #"log_avg_num_partners_contact_parent", "log_avg_num_parents_contact_partner_2"
                            color_continuous_scale="Turbo_r", 
                            render_mode="svg",
                            labels={ "log_ratio": "log10 (# Parent CT / # Partner CT)", "log_avg_num_partners_contact_parent": "Log Avg # of Partners a Parent Contacts (1)", "log_avg_num_parents_contact_partner_2": "Log Avg # of Parents a Partner Contacts (2)"})
        fig.update_layout(
            showlegend = False,
            font=dict(
                family="Arial",
                size=16),
            # title={
            #     'text': f"#Parent / #Partner Log Ratio Plotted with Partner's Contact Pattern: {direction}]"},        
            plot_bgcolor='white', paper_bgcolor='white',
            width=1400,
            height=900)

    fig.update_xaxes(
        title_text=None
    )

    fig.update_yaxes(
        title_text=None
    )
    fig.update_coloraxes(showscale=False)
    return fig

# Two-way reliability: Plot synapases for two connected neurons (this was to toruble shoot the Pm2a Mi1, Mi9, Mi2, Mi4 reliability asymmetry) 

def get_syn_position(parent, partner, roi):
    """
    Get the average presynaptic position of parent neurons and postsynaptic
    position of partner neurons involved in a parent → partner connection.

    Parameters
    ----------
    parent : str
    partner : str
    roi : str or list
        Example: ['ME(R)', 'AME(R)', 'LO(R)', 'LOP(R)']

    Returns
    -------
    df_celltype_fin
        Mean synapse positions with reliability labels.

    df_freq_bodyId_conn
        Number of times each neuron participates in the connection.
    """
    cfg = pre_HP.configpath()
    data_dir = Path(cfg["data_dir"])
    partner_folder = "partner_data_28_04_2026"

    if isinstance(roi, str):
        roi = [roi]

    # Load parent- and partner-centered data
    filepath_in = data_dir / partner_folder / f"input_{partner}_df.pkl"
    with open(filepath_in, "rb") as f:
        input_partner = pickle.load(f)

    filepath_out = data_dir / partner_folder / f"output_{parent}_df.pkl"
    with open(filepath_out, "rb") as f:
        output_parent = pickle.load(f)

    input_partner = input_partner[input_partner["type_pre"] == parent] # partner reliability
    output_parent = output_parent[output_parent["type_post"] == partner] # parent reliability 

    # Parent presynaptic positions
    df_syn = fetch_synapses(
        NC(instance=f"^{parent}(?![0-9A-Za-z]).*_R$", regex=True, status="Traced"),
        SC(rois=roi, primary_only=True)
    )

    df_syn = df_syn[(df_syn["confidence"] > 0.5) & (df_syn["type"] == "pre")]
    df_syn = df_syn.drop_duplicates(subset=["bodyId", "type", "x", "y", "z"])

    df_parent = df_syn.groupby(["bodyId", "type"], as_index=False)[["x", "y", "z"]].mean()
    df_parent["reliability"] = df_parent["bodyId"].isin(output_parent["bodyId_pre"]).astype(int)
    df_parent.insert(2, "name", parent)
    df_parent["connection_role"] = "parent"

    # Partner postsynaptic positions
    df_syn = fetch_synapses(
        NC(instance=f"^{partner}(?![0-9A-Za-z]).*_R$", regex=True, status="Traced"),
        SC(rois=roi, primary_only=True)
    )

    df_syn = df_syn[(df_syn["confidence"] > 0.5) & (df_syn["type"] == "post")]
    df_syn = df_syn.drop_duplicates(subset=["bodyId", "type", "x", "y", "z"])

    df_partner = df_syn.groupby(["bodyId", "type"], as_index=False)[["x", "y", "z"]].mean()
    df_partner["reliability"] = df_partner["bodyId"].isin(input_partner["bodyId_post"]).astype(int)
    df_partner.insert(2, "name", partner)
    df_partner["connection_role"] = "partner"

    df_celltype_fin = pd.concat([df_parent, df_partner], ignore_index=True)

    # Number of times each neuron participates in the connection
    parent_frequency = output_parent["bodyId_pre"].value_counts().rename_axis("bodyId").reset_index(name="count")
    parent_frequency["connection_role"] = "parent"
    parent_frequency["name"] = parent
    parent_frequency["type"] = "pre"

    partner_frequency = input_partner["bodyId_post"].value_counts().rename_axis("bodyId").reset_index(name="count")
    partner_frequency["connection_role"] = "partner"
    partner_frequency["name"] = partner
    partner_frequency["type"] = "post"

    df_freq_bodyId_conn = pd.concat([parent_frequency, partner_frequency], ignore_index=True)

    return df_celltype_fin, df_freq_bodyId_conn
def plot_mean_syn(parent, partner, df_celltype, df_freq_bodyId_conn):
    """
    Plot the mean presynaptic positions of parent neurons and mean
    postsynaptic positions of partner neurons.

    Connected neurons are coloured by the number of connection records.
    Unconnected neurons are shown in grey.
    """

    parent_df = df_celltype[(df_celltype['type'] == 'pre') & (df_celltype['name'] == parent)]
    partner_df = df_celltype[(df_celltype['type'] == 'post') & (df_celltype['name'] == partner)]

    df_conn_parent = parent_df[parent_df['reliability'] == 1]
    df_not_parent = parent_df[parent_df['reliability'] == 0]

    df_conn_partner = partner_df[partner_df['reliability'] == 1]
    df_not_partner = partner_df[partner_df['reliability'] == 0]

    # Colour neurons by frequency of occurrence in the connection table
    parent_color = df_freq_bodyId_conn[(df_freq_bodyId_conn['connection_role'] == 'parent') & (df_freq_bodyId_conn['type'] == 'pre')]
    partner_color = df_freq_bodyId_conn[(df_freq_bodyId_conn['connection_role'] == 'partner') & (df_freq_bodyId_conn['type'] == 'post')]

    df_conn_parent = df_conn_parent.merge(parent_color[['bodyId', 'count']], on='bodyId', how='left')
    df_conn_partner = df_conn_partner.merge(partner_color[['bodyId', 'count']], on='bodyId', how='left')

    df_conn_parent['count'] = df_conn_parent['count'].fillna(0)
    df_conn_partner['count'] = df_conn_partner['count'].fillna(0)

    parent_cmax = max(df_conn_parent['count'].max(), 1)
    partner_cmax = max(df_conn_partner['count'].max(), 1)

    fig = go.Figure()

    # Connected parent neurons
    fig.add_trace(go.Scatter3d(
        x=df_conn_parent['x'],
        y=df_conn_parent['y'],
        z=df_conn_parent['z'],
        mode='markers',
        name=f"{parent}_connected",
        marker=dict(
            size=5,
            opacity=0.9,
            color='Blue'
            # color=df_conn_parent['count'],
            # colorscale='Blues',
            # cmin=0,
            # cmax=parent_cmax,
            # colorbar=dict(title="Parent frequency", x=1.05)
        ),
        text=df_conn_parent['bodyId'],
        customdata=df_conn_parent['count'],
        hovertemplate="bodyId: %{text}<br>frequency: %{customdata}<extra></extra>"
    ))

    # Unconnected parent neurons
    fig.add_trace(go.Scatter3d(
        x=df_not_parent['x'],
        y=df_not_parent['y'],
        z=df_not_parent['z'],
        mode='markers',
        name=f"{parent}_not_connected",
        marker=dict(size=5, color='lightgrey', opacity=0.2, line=dict(color='black', width=1)),
        text=df_not_parent['bodyId'],
        hovertemplate="bodyId: %{text}<extra></extra>"
    ))

    # Connected partner neurons
    fig.add_trace(go.Scatter3d(
        x=df_conn_partner['x'],
        y=df_conn_partner['y'],
        z=df_conn_partner['z'],
        mode='markers',
        name=f"{partner}_connected",
        marker=dict(
            size=5,
            opacity=0.9,
            color='Red'
            # color=df_conn_partner['count'],
            # colorscale='Reds',
            # cmin=0,
            # cmax=partner_cmax,
            # colorbar=dict(title="Partner frequency", x=-0.15)
        ),
        text=df_conn_partner['bodyId'],
        customdata=df_conn_partner['count'],
        hovertemplate="bodyId: %{text}<br>frequency: %{customdata}<extra></extra>"
    ))

    # Unconnected partner neurons
    fig.add_trace(go.Scatter3d(
        x=df_not_partner['x'],
        y=df_not_partner['y'],
        z=df_not_partner['z'],
        mode='markers',
        name=f"{partner}_not_connected",
        marker=dict(size=5, color='lightgrey', opacity=0.2, line=dict(color='black', width=1)),
        text=df_not_partner['bodyId'],
        hovertemplate="bodyId: %{text}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Parent: {parent} → Partner: {partner}",
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(
                x=1,
                y=1,
                z=1
            ),
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(x=-0.25, y=1, xanchor='left', yanchor='top'),
        margin=dict(l=150)
    )
    fig.show(config={
    "modeBarButtonsToAdd": ["toImage"],
    "toImageButtonOptions": {
        "format": "svg",
        "filename": f"{parent}_{partner}"
        }
    })

    return fig