#  Main for Temporal Data Creation:
import os
import torch
import pickle
import logging
import networkx
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

# ------- custom modules -------
from preprocessing.pytorch_dataloader import DatasetLoader

# ======================================================================================================================
#   Global Variables
# ======================================================================================================================

# ======================================================================================================================
#   Preparation Functions
# ======================================================================================================================

def transform_df_to_npy(timeseries: pd.DataFrame, number_of_nodes: int, save: bool = False) -> np.array:
    """
    This function transform an ORDERED IN ASCENDING ORDER BY TIME timeseries df to .npy 3D.
    Also, it arranges in order the nodes to be sorted in the same way for each snapshot

    Parameters
    ----------
    timeseries
    number_of_nodes
    save

    Returns
    -------
    node_values
    """
    node_number = number_of_nodes
    slice_temp = []
    node_values_list = []
    count = 0
    for i, row in timeseries.iterrows():
        if count == node_number:
            slice_temp_df = pd.DataFrame(slice_temp)
            slice_temp_df.sort_values(by=[1], inplace=True)
            slice_temp_lmp = slice_temp_df[0]
            slice_temp_tuple = tuple(slice_temp_lmp.values.tolist())
            node_values_list.append(slice_temp_tuple)
            slice_temp = []
            count = 0
            values = row[["LMP", "nodeID"]]
            edgeList = values.values.tolist()
            slice_temp.append(edgeList)
        else:
            values = row[["LMP", "nodeID"]]
            edgeList = tuple(values.values.tolist())
            slice_temp.append(edgeList)
        count += 1
    node_values_ercot = np.array(node_values_list)

    if save:
        with open('data/nodes_values.npy', 'wb') as f:
            np.save(f, node_values_ercot)

    return node_values_ercot




# ======================================================================================================================
#   Analytical Functions
# ======================================================================================================================

def get_adj_matrix(dependencies: pd.DataFrame, save: bool = False) -> np.array:
    """
    This function create the network of the model and extract the adj matrix.
    Parameters
    ----------
    dependencies df: ["zoneID", "depends_from_ID", "dependence_type"]
    save:

    Returns
    -------
    A
    """
    weighted_dependencies = dependencies[["zoneID", "depends_from_ID", "dependence_type"]]
    edge_list = weighted_dependencies.values.tolist()
    G = networkx.Graph()

    for i in range(len(edge_list)):
        G.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    A0 = networkx.adjacency_matrix(G).A

    # Weights normalization:
    A0 = abs(A0)
    row_sums = A0.sum(axis=1)
    A = A0 / row_sums[:, np.newaxis]

    if save:
        with open('data/A.npy', 'wb') as f:
            np.save(f, A)
    return A


# ======================================================================================================================
#  Debug Entrypoint
# ======================================================================================================================
if __name__ == '__main__':
    # local variables
    PROJECT_ROOT: Path = Path(__file__).parents[1]  # root folder of the repository
    DATA_LOCATION: Path = PROJECT_ROOT / 'data'
    RAW_DATA_LOCATION: Path = DATA_LOCATION / 'raw'
    #PREPROC_DATA_LOCATION: Path = DATA_LOCATION / 'preprocessed'  # root folder of the repository #

    # 0 - Graph Info Required:
    number_of_nodes: int = 36

    # 1 - Forecasting specifications:
    time_window = 4 # 60 minutes (15 min of sample time)
    forecasting_window  = 1

    # 2 - Transform the timeseries from df to numpy:
    timeseries_df = pd.read_csv("./data/timeseries_one_year.csv")  # select manually the dataframe for your experiments
    node_values_ercot = transform_df_to_npy(timeseries=timeseries_df, number_of_nodes=number_of_nodes, save=False)

    # 3 - Get the adj matrix from dependencies' df:
    dependencies_df = pd.read_csv('./data/dependencies.csv')
    adj_matrix = get_adj_matrix(dependencies=dependencies_df, save=False)

    # 4 - Get the temporal tGNN using Pytorch Class:
    loader = DatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=time_window, num_timesteps_out=forecasting_window, save=False)

    # 5 - Model Training and Evaluation:
    # TODO: read learned parameter

    # 5 - Prediction:
    # TODO: add prediction step

    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))
    print(next(iter(dataset)))

    logging.info('-----> done <-----')
