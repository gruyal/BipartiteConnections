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
import Handshake_pair_v2
import yaml
import os
import openpyxl
from pathlib import Path
import pickle


def main():
    x_cutoff = 3
    y_cutoff = 0.8
    input_mono_file = "input_monopartite_table_v3.pkl"
    output_mono_file = "output_monopartite_table_v3.pkl"
    input_bipartite = "bipartite_input_parent_output_partner_v3.pkl"
    output_bipartite = "bipartite_output_parent_input_partner_v3.pkl"
    partner_folder = "partner_data_28_04_2026"

    cfg = pre_HP.configpath()
    data_dir = Path(cfg['data_dir'])

    save_dir = data_dir / partner_folder

    client = Client(
        f"https://{cfg['client']}",
        dataset=cfg['dataset'],
        token=cfg['token']
    )

    # Load optic lobe cell types
    all_ol_ct = pd.read_excel(data_dir / "supplement_table" / "Sup_Table_1_Cell-types_and_counts_final.xlsx")
    all_ol_ct_R = all_ol_ct[all_ol_ct["instance"] != all_ol_ct["cell_type"] + "_L"].reset_index(drop=True)
    all_celltypes_R_list = (all_ol_ct_R[['cell_type', 'Group']].drop_duplicates().set_index('cell_type')['Group'].to_dict()) # all cts 
    all_celltypes_list_greater_or_eq_10 = (all_ol_ct_R.loc[all_ol_ct_R['no. of cells'] >= 10,['cell_type', 'Group']].drop_duplicates().set_index('cell_type')['Group'] .to_dict())

    # print("Fetching all OL data...")
    # # create directory if it doesn't exist
    # save_dir.mkdir(parents=True, exist_ok=True)
    # # load all OL celltypes! 
    # for i in  all_celltypes_R_list:
    #     og_input_to_celltype_df = pre_HP.get_input_connections(i, 'right')
    #     og_output_to_celltype_df = pre_HP.get_output_connections(i, 'right')

    #     # save input
    #     with open(save_dir / f"input_{i}_df.pkl", "wb") as f:
    #         pickle.dump(og_input_to_celltype_df, f)

    #     # save output
    #     with open(save_dir / f"output_{i}_df.pkl", "wb") as f:
    #         pickle.dump(og_output_to_celltype_df, f)

    # # Run bipartite analysis for all_celltypes_list_greater_or_eq_10

    print("Running monopartite population analysis...")

    input_monopartite_table_v3, output_monopartite_table_v3 = \
        Handshake_pair_v2.monopartite_population_first(all_celltypes_list_greater_or_eq_10, x_cutoff, y_cutoff)

    print("Saving monopartite results...")

    with open(data_dir /input_mono_file, "wb") as f:
        pickle.dump(input_monopartite_table_v3, f)

    with open(data_dir / output_mono_file, "wb") as f:
        pickle.dump(output_monopartite_table_v3, f)

    # print("Querying partner neurons...")
    # Handshake_pair_v2.query_partners(
    #     input_monopartite_table_v3,
    #     output_monopartite_table_v3,
    #     all_ol_ct,
    #     all_celltypes_R_list
    # )
    # failure_file = data_dir / partner_folder / "failed_neurons.txt"
    # neurons = pd.read_csv(failure_file, header=None)[0].str.strip().tolist()

    # for i in failure_file:
    #     og_input_to_celltype_df = pre_HP.get_input_connections(i, 'middle')

    #     with open(data_dir / partner_folder/ ("input_" + i + "_df.pkl"), "wb") as f:
    #         pickle.dump(og_input_to_celltype_df, f)

    print("Running bipartite population analysis...")

    bipartite_parent_output_v3, bipartite_parent_input_v3 = Handshake_pair_v2.bipartite_population_second(input_monopartite_table_v3, output_monopartite_table_v3)

    with open(data_dir / output_bipartite, "wb") as f:
        pickle.dump(bipartite_parent_output_v3, f)

    with open(data_dir / input_bipartite, "wb") as f:
        pickle.dump(bipartite_parent_input_v3, f)


    print("Saving bipartite population  results...")


    # These bipartite pair are for all parent celltypes incuding those that did not quality the >= 10 neuron threshold 

    all_celltypes_R_list
    print("Running monopartite population analysis...")

    all_input_monopartite_table_v3, all_output_monopartite_table_v3 = \
        Handshake_pair_v2.monopartite_population_first(all_celltypes_R_list, x_cutoff, y_cutoff)

    print("Saving monopartite results...")

    with open(data_dir /"all_OL_input_monopartite_table_v3.pkl", "wb") as f:
        pickle.dump(all_input_monopartite_table_v3, f)

    with open(data_dir / "all_OL_output_monopartite_table_v3.pkl", "wb") as f:
        pickle.dump(all_output_monopartite_table_v3, f)

    print("Saving bipartite population  results...")

if __name__ == "__main__":
    main()