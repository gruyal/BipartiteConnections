import pandas as pd
from kneed import KneeLocator
from neuprint import Client
from neuprint import fetch_adjacencies, NeuronCriteria as NC, fetch_synapses, SynapseCriteria as SC, fetch_neurons
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pre_HP
import Handshake_function_v1

def compute_coordinates(parent_list, direction_parent):
    """
    Compute Cartesian coordinates for plotting partners of a parent cell.
    Places partners evenly along a semicircle depending on direction. input is above x axis and output is below y axis 
    
    Args:
        parent_list (list): Partners of the parent cell.
        direction_parent (str): 'input' or 'output'.
    
    Returns:
        dict: {partner_name: (x, y) coordinates}
    """
    # Divide angles across semi-circle: [pi/6, 5pi/6] → for (inputs), or [11pi/6, 7pi/6] for (outputs)
    radius = 10
    coordinates = {}
    num = len(parent_list)
    if direction_parent == 'input':
        angles = np.linspace(5*np.pi/4, 3*np.pi/4, num, endpoint=True)

    elif direction_parent == 'output':
        angles = np.linspace(-1*np.pi/4, np.pi/4, num, endpoint=True)

    elif direction_parent == 'both':
        angles = np.linspace(np.pi/3, 2*np.pi/3, num, endpoint=True)
    
    tuple_angle_partner = zip(parent_list, angles)

    # cos is x and sin is y for each partner
    for partner, angle in tuple_angle_partner:
        coordinates[partner, direction_parent] = (radius * np.cos(angle), radius * np.sin(angle))

    return coordinates 

def coordinates_bipartite_population(parent_cell, bipartite_parent_input, bipartite_parent_output):
   """
    Get coordinates for ploltly go scatter plot for arrows and cells.

    Parameters:
    - parent_cell: cell of interest. ONLY ONE CELL.
    - bipartite_input_parent_output_partner : dictionary of handshake information for a parent type. Input to parent, output from partner
    - bipartite_output_parent_input_partner : dictionary of handshake information for a parent type. output from parent, input to partner

    Return: 
    - midpoint_coordinates, node_position (coordinates for cell and arrows)

    """
 
   parent_in_list_orig = bipartite_parent_input[bipartite_parent_input["parent"] == parent_cell]['partner_cell'].to_list()
   parent_out_list_orig = bipartite_parent_output[bipartite_parent_output["parent"] == parent_cell]['partner_cell'].to_list()
   
   # 3 lists: ct present in only input or output and list for present in both (set is unordered, use list comprehension
   parent_in_out_list = [x for x in parent_in_list_orig if x in parent_out_list_orig]  # intersection
   parent_in_list = [x for x in parent_in_list_orig if x not in parent_out_list_orig]  # only input
   parent_out_list = [x for x in parent_out_list_orig if x not in parent_in_list_orig]  # only output


   # Parent node position 
   node_position = { (parent_cell, 'center'): (0, 0) }
   
   coordinates_input = compute_coordinates(parent_in_list, 'input')
   coordinates_output = compute_coordinates(parent_out_list, 'output')
   coordinates_output_input = compute_coordinates(parent_in_out_list, 'both')

   # Update node positions for partner cells 
   node_position.update(coordinates_input)
   node_position.update(coordinates_output)
   node_position.update(coordinates_output_input)

   midpoint_coordinates = {}
   for key, (x,y) in node_position.items():
      # Get coordinates for partner and compute midpoint for y value (arrows)
      midpoint_coordinates[key] = (x / 2, y / 2)

   return midpoint_coordinates, node_position
# Final Plot 

def plot_bipartite_population(parent_cell, bipartite_parent_input, bipartite_parent_output):
    """
    Plot bipartite circuit diagram for population. Parent cell input/output --> partner cell and its strength with consideration of elbows to take into account. 

    Parameters:
    - parent_cell: cell of interest. ONLY ONE CELL.
    - bipartite_input_parent_output_partner : dictionary of handshake information for a parent type. Input to parent, output from partner
    - bipartite_output_parent_input_partner : dictionary of handshake information for a parent type. output from parent, input to partner

    """
    
    # Helper: width calculation
    def width_from_pcnt(pcnt):
        if pd.isna(pcnt):
            return 0 
        else:
            return max(pcnt , 1)

    # Helper: add line trace
    def add_line(x0, y0, x1, y1, width, color):
        plot.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=width, color=color),
                mode="lines",
                showlegend=False
            )
        )

    # Helper: add annotation
    def add_annotation(x, y, text, color, anchor_x, anchor_y):
        plot.add_annotation(
            x=x, y=y,
            text=text,
            font=dict(color=color, size=30),
            xanchor=anchor_x,
            yanchor=anchor_y
        )

    # --- Main plotting ---
    midpoint_coordinates, node_position = coordinates_bipartite_population(parent_cell, bipartite_parent_input, bipartite_parent_output)

    plot = go.Figure()

    # Plot connections
    first_val = bipartite_parent_input[bipartite_parent_input["parent"] == parent_cell]
    if first_val.empty:
        return None   # skip entire plot
    first_val = first_val["num_neurons_parent_ct"].iloc[0]

    for (cell_name, direction_type), (x_mid, y_mid) in midpoint_coordinates.items():
        if direction_type == "center":
            add_annotation(0, -3, f"<b>Focal Cell-Type: {cell_name}<br>Num Neurons: {first_val}</b>", '#762a83' , "center", "bottom")
            continue

        if direction_type == "input":
            row_in = bipartite_parent_input[(bipartite_parent_input["parent"] == parent_cell) & (bipartite_parent_input["partner_cell"] == cell_name)]
            if row_in.empty:
                continue

            row_in = row_in.iloc[0]
            add_line(x_mid, y_mid, -0.1, 0, width_from_pcnt(row_in["pcnt_input_to_parent"]), '#1b7837')  # input to parent
            add_line(x_mid*2, y_mid*2, x_mid, y_mid, width_from_pcnt(row_in["pcnt_output_from_partner_2"]), '#3288bd')  # output from partner

        # [partner rank: parent rank]
            add_annotation(
            x_mid*2.05, y_mid*2,
            f"<b>{cell_name}</b><br>",
            # f"<b>{cell_name}[{row_in['partner_rank_2']}:{row_in['parent_rank']}] </b><br>",
            "#1b7837", "right", "top")
                
        if direction_type == "output":
            row_out = bipartite_parent_output[(bipartite_parent_output["parent"] == parent_cell) & (bipartite_parent_output["partner_cell"] == cell_name)]
            if row_out.empty:
                continue

            row_out = row_out.iloc[0]
            add_line(0.1, 0, x_mid, y_mid, width_from_pcnt(row_out["pcnt_output_from_parent"]), '#3288bd')  # output from parent
            add_line(x_mid, y_mid, x_mid*2, y_mid*2, width_from_pcnt(row_out['pcnt_input_to_partner_2']), '#1b7837')  # input to partner

            # [parent rank: partner rank]
            add_annotation(
            x_mid*2.1, y_mid*2,
            f"<b>{cell_name}</b><br>",
            # f"<b>{cell_name}[{row_out['partner_rank_2']}:{row_out['parent_rank']}] </b><br>",
            '#3288bd', "left", "top")

        if direction_type == "both":
            row_out = bipartite_parent_output[(bipartite_parent_output["parent"] == parent_cell) & (bipartite_parent_output["partner_cell"] == cell_name)]
            row_in = bipartite_parent_input[(bipartite_parent_input["parent"] == parent_cell) & (bipartite_parent_input["partner_cell"] == cell_name)]
            if row_out.empty or row_in.empty:
                continue

            row_out = row_out.iloc[0]
            row_in = row_in.iloc[0]

        # Elbow proportional shift 
            elbow_shift = 0.3
            elbow_in_x = x_mid - elbow_shift
            elbow_in_y = y_mid + elbow_shift

            elbow_out_x = x_mid + elbow_shift
            elbow_out_y = y_mid + elbow_shift

            # Partner → Parent (Parent Input)
            add_line(x_mid*2, y_mid*2, elbow_out_x, elbow_out_y, width_from_pcnt(row_in["pcnt_output_from_partner_2"]), '#3288bd')  # output from partner
            add_line(elbow_out_x, elbow_out_y, 0, 0.1, width_from_pcnt(row_in["pcnt_input_to_parent"]), '#1b7837')  # input to parent

            # Parent → Partner (Parent Output)
            add_line(0.1, 0, elbow_in_x, elbow_in_y, width_from_pcnt(row_out["pcnt_output_from_parent"]), "#91bfdb")  # output from parent
            add_line(elbow_in_x, elbow_in_y, x_mid*2, y_mid*2, width_from_pcnt(row_out["pcnt_input_to_partner_2"]), "#a6dba0")  # input to partner

            # [parent rank: partner rank]
            add_annotation(
                    x_mid*2.05, y_mid*2.1,
                    f"<b>{cell_name}</b><br>",
                    # f"<b>{cell_name}[{row_in['partner_rank_2']}:{row_in['parent_rank']}] </b><br>",
                    '#1b7837', "center", "top")

            # # [parent rank: partner rank]
            # add_annotation(
            #         x_mid*2, y_mid*2.1,
            #         f"<b>{cell_name}[{row_out['partner_rank_2']}:{row_out['parent_rank']}] </b><br>",
            #         "#91bfdb", "center", "top")
            
    # --- Add thickness legend traces ---
    # # pick 4 reference points in your scale (could be linear or quantiles)
    # thickness_examples = [
    #     ("1%", 2),
    #     ("5%", 5),
    #     ("10%", 10),
    #     ("20%", 20),
    #     ("30%", 40),
    #     ("60%", 60),
    # ]
    # # Anchor corner for scale bar
    # x0, y0 = -1, -3  
    # x , y = 1 , -3

    # for i, (label, width) in enumerate(thickness_examples):
    #     dy = i * 0.7
    #     # Draw scale line using your helper
    #     add_line(x0, y0 - dy, x, y - dy, width, "black")
        
    #     # Add annotation (label)
    #     add_annotation(
    #         x0 - 0.5,  
    #         y - dy,
    #         f"{label}",
    #         "black",
    #         "center",
    #         "middle"
    #     )

    # --- Update layout for legend ---
    plot.update_layout(
        width=3000,    # width in pixels
        height=3000,   # height in pixels
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(scaleanchor="x", scaleratio=1, showticklabels=False, showline=False, showgrid=False, zeroline=False),
        xaxis=dict(constrain="domain", showticklabels=False, showline=False, showgrid=False, zeroline=False),
        title=dict(
            text=f"Bipartite Relationship between Parent Cell {parent_cell} and its Partner Cells",
            x=0.5,  # center title
            xanchor="center"
        ),
        legend=dict(
            orientation="h",      
            y=0.3,                    
            x=0.49          
        )
    )
        # Plot partner nodes
    for (cell_name, direction_type), (x, y) in node_position.items():
        
        if direction_type == 'input':
            col = '#1b7837'

        elif direction_type == 'output':
            col = '#3288bd'
            
        elif direction_type == 'both': 
            col = '#5ab4ac'

        elif direction_type == 'center':
            col = '#762a83'
        
        plot.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                name=cell_name + "<br>" + direction_type,
                marker=dict(size=50, color=col),
                legendgroup=direction_type,
                showlegend=False
            )
        )
        
        if direction_type in ['input', 'both']:
            row_in = bipartite_parent_input[(bipartite_parent_input["parent"] == parent_cell) & (bipartite_parent_input["partner_cell"] == cell_name)]
            if not row_in.empty:
                row_in = row_in.iloc[0]
                if 1 <= row_in['partner_rank_2'] <= 5:
                    plot.add_trace(go.Scatter( x=[x], y=[y], mode="markers", marker=dict(size=60, color="rgba(0,0,0,0)", line=dict(color="red", width=3)), showlegend=False))

        if direction_type in ['output', 'both']:
            row_out = bipartite_parent_output[(bipartite_parent_output["parent"] == parent_cell) &(bipartite_parent_output["partner_cell"] == cell_name)]
            if not row_out.empty:
                row_out = row_out.iloc[0]
                if 1 <= row_out['partner_rank_2'] <= 5:
                            plot.add_trace(go.Scatter( x=[x], y=[y], mode="markers", marker=dict(color="black", size=10), showlegend=False))
    
    return plot


