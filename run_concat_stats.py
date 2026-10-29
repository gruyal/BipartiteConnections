import pandas as pd
from neuprint import Client
from neuprint import fetch_adjacencies, NeuronCriteria as NC, fetch_synapses, SynapseCriteria as SC, fetch_neurons
import plotly.express as px
import numpy as np
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pre_HP
import elbow_functions
import yaml
import os
import openpyxl
from pathlib import Path
from kneed import KneeLocator
import cProfile
import pstats
import pickle

def main():
    print("STARTED run_concat_stats.py")
    cfg = pre_HP.configpath()

    data_dir = Path(cfg['data_dir'])
    elbow_related = Path(cfg['elbow_related'])
    concat_related = Path(cfg['concat_related'])

    client = Client(
        f"https://{cfg['client']}",
        dataset=cfg['dataset'],
        token=cfg['token']
    )

    # Load ALL celltypes and cts >= 10
    all_ol_ct = pd.read_excel(data_dir/"supplement_table"/"Sup_Table_1_Cell-types_and_counts_final.xlsx")
    all_ol_ct_R = all_ol_ct[all_ol_ct["instance"] != all_ol_ct["cell_type"] + "_L"].reset_index(drop=True) # only right side
    all_celltypes_list = (all_ol_ct_R[['cell_type', 'Group']].drop_duplicates().set_index('cell_type')['Group'].to_dict()) # all cts 
    all_celltypes_list_greater_or_eq_10 = (all_ol_ct_R.loc[all_ol_ct_R['no. of cells'] >= 10, ['cell_type', 'Group']].drop_duplicates().set_index('cell_type')['Group'].to_dict()) # filtered >= 10 this criteria removed all non-optic lobe neurons (VPN, ONCN, ONIN, VCN)

    # All NTs
    NT_all_cts = pd.read_csv(data_dir/ "NT_all_cts.csv")

    # elbow_stats_concat_pop_new = elbow_functions.get_all_parent_elbows(all_celltypes_list_greater_or_eq_10, 'population', 'partner_dist') 
    # elbow_stats_concat_pop_new.to_csv(elbow_related / "elbow_stats_concat_pop_new.csv", index=False)

    # elbow_stats_concat_indv_new = elbow_functions.get_all_parent_elbows(all_celltypes_list_greater_or_eq_10, 'individual', 'partner_dist') 
    # elbow_stats_concat_indv_new.to_csv(elbow_related / "elbow_stats_concat_indv_new.csv", index=False)

    # # These are ranked via weighted median

    # concat_top_input_output = pre_HP.get_all_parent_top_cts(all_celltypes_list_greater_or_eq_10 , 1)
    # concat_top_input_output.to_csv(concat_related  / "concat_top_input_output.csv", index=False)

    concat_top_input_output_10_weight_median = pre_HP.get_all_parent_top_cts(all_celltypes_list_greater_or_eq_10 , 10)
    concat_top_input_output_10_weight_median.to_csv(concat_related  / "concat_top_input_output_10_weight_median.csv", index=False)

    # concat_top_input_output_5_weight_median
    concat_top_input_output_5_weight_median = concat_top_input_output_10_weight_median.copy()
    concat_top_input_output_5_weight_median = concat_top_input_output_5_weight_median[concat_top_input_output_5_weight_median['weighted_median_rank'].between(1, 5)]
    concat_top_input_output_5_weight_median.to_csv(concat_related  / "concat_top_input_output_5_weight_median.csv", index=False)

    print("FINISHED run_concat_stats.py", flush=True)

if __name__ == "__main__":
    main()